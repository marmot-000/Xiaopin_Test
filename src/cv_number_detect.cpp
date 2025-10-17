#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>

using namespace cv;
using namespace std;
int temp_num = 27;  // 模板数量
Mat mask_1, mask_2, red_mask;  // 掩膜
const string DOCKER_TEMPLATES_DIR = "/opt/ep_ws/src/rmus_solution/temple/";  // 模板路径（容器内）
const string LOCAL_TEMPLATES_DIR = "/home/robot/图片/temple/";  // 模板路径（宿主机）
vector<Mat> num_templates;  // 存储模板
struct RectangleData 
{
    int id;  //方块id
    vector<Point2f> corners;  // 方块的4个角点
};

class CubeNumberRecognizer
{
public:
    void processImage(Mat img_original);  // 图像预处理
    vector<RectangleData> getCornerPoints();  // 角点识别及排序
    vector<Mat> perspecTransform(const vector<RectangleData>& rect_contours, int target_size, Mat input_image); 
    vector<Mat> loadNumberTemplates(int temp_num, const string templates_dir, int target_size);  // 加载数字模板
    vector<int> getDectedNumber(const vector<Mat> images_input, vector<Mat>& templates, int target_size);  
    void visualizeResults(Mat& img, const vector<RectangleData>& rectangles);  // 可视化标注
    Mat jointCanvas(int target_size, vector<Mat> warped_images); // 拼接多个方块透视的图
private:
    
};


vector<Mat> CubeNumberRecognizer::loadNumberTemplates(int temp_num, const string templates_dir, int target_size)
{
    vector<Mat> num_templates;
    for(int i = 0; i < temp_num; i++)
    {
        string path = templates_dir + to_string(i + 1) + ".png";
        Mat temp = imread(path, IMREAD_GRAYSCALE);
        if (temp.empty())
        {
            ROS_ERROR("Temple file can not found: %s", path.c_str());
            return num_templates;  // 修正：返回空向量，避免无返回值
        }
        resize(temp, temp, Size(target_size, target_size));
        num_templates.push_back(temp);
    }
    ROS_INFO("successfully loading templates, sum:%d", (int)num_templates.size());
    return num_templates;
}


vector<int> CubeNumberRecognizer::getDectedNumber(const vector<Mat> images_input, vector<Mat>& templates, int target_size)
{
    vector<int> number;
    Mat digital_binary, show_img; 
    for (auto& p : images_input) 
    { 
        double min_val = 1e9;
        int best_match_idx = -1;
        Scalar lower_digital = Scalar(85, 85, 95); // 数字BGR最小值
        Scalar upper_digital = Scalar(105, 105, 105); // 数字BGR最大值
        inRange(p, lower_digital, upper_digital, digital_binary);//提取数字二值图
        
        Mat kernel = getStructuringElement(MORPH_RECT, Size(2, 2)); 
        morphologyEx(digital_binary, digital_binary, MORPH_OPEN, kernel); 
        resize(digital_binary, digital_binary, Size(target_size, target_size));
        show_img = digital_binary; // 用于后续显示验证
        
        for(int i = 0; i < templates.size(); i++)
        {
            Mat result;
            matchTemplate(digital_binary, templates[i], result, TM_SQDIFF);
            double val;
            minMaxLoc(result, &val, nullptr, nullptr, nullptr);
        
            if (val < min_val)
            {
                min_val = val;
                best_match_idx = i;
            } 
        }
       
        number.push_back(((min_val < 5e7) ? ((best_match_idx + 1)/4 + 1) : -1)); // 4个模板对应1个数字，计算实际数字
    }
    return number;
}

// 修正：改为类成员函数，修正颜色转换宏和闭运算类型
void CubeNumberRecognizer::processImage(Mat img_original)
{
    //1.将RGB转换成HSV，便于截取（修正：CV_RGB2HSV → COLOR_RGB2HSV）
    Mat img_hsv;
    cvtColor(img_original, img_hsv, COLOR_RGB2HSV);
    //2.建两个掩膜（修正：mask_2范围应为160-179，避免与mask_1重复）
    inRange(img_hsv, Scalar(0, 112, 70), Scalar(10, 255, 255), mask_1);
    inRange(img_hsv, Scalar(160, 112, 70), Scalar(179, 255, 255), mask_2); // 修正：红色另一范围
    bitwise_or(mask_1, mask_2, red_mask);
    //3.开运算消除噪点
    Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(red_mask, red_mask, MORPH_OPEN, element);
    //4.闭运算连通孔洞（修正：MORPH_OPEN → MORPH_CLOSE）
    morphologyEx(red_mask, red_mask, MORPH_CLOSE, element);
}


vector<RectangleData> CubeNumberRecognizer::getCornerPoints()
{
    int current_id = 1;
    vector<RectangleData> rect_contours;
    //1.识别所有轮廓
    vector<vector<Point>> contours;
    Mat mask_copy = red_mask.clone();
    findContours(mask_copy, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);//描边
    //2.遍历所有轮廓找到符合要求的四个点的多边形轮廓
    for(auto &contour : contours)
    {
        //过滤掉太小的轮廓
        double area = contourArea(contour);
        if(area < 500) continue;
        //轮廓周长计算
        double perimeter = arcLength(contour, true);
        vector<Point> approx; // 存储近似多边形
        approxPolyDP(contour, approx, 0.02 * perimeter, true); 
        Point2f center(0, 0);
        if(approx.size() == 4)
        {
            //排序，给id
            RectangleData rect;
            rect.id = current_id++; 
            for(auto& p : approx) 
            {
                rect.corners.push_back(Point2f(p.x, p.y));
                center.x += p.x;
                center.y += p.y;
            }
            center /= 4;
            // 角点按角度排序
            sort(rect.corners.begin(), rect.corners.end(), [center](const Point2f& a, const Point2f& b){
                double atan2_a = atan2(a.y - center.y, a.x - center.x);
                double atan2_b = atan2(b.y - center.y, b.x - center.x);
                return atan2_a < atan2_b;
            });
            rect_contours.push_back(rect);
        }
    }
    return rect_contours;
}


vector<Mat> CubeNumberRecognizer::perspecTransform(const vector<RectangleData>& rect_contours, int target_size, Mat input_image)
{
    vector<Mat> warped_images;
    //1.透视变换目标坐标
    vector<Point2f> dst = { Point2f(0,0), Point2f(target_size,0), Point2f(target_size,target_size), Point2f(0,target_size) };
    for(auto& rect_contour : rect_contours)
    {
        //计算透视变换矩阵
        Mat M = getPerspectiveTransform(rect_contour.corners, dst);
        Mat warped_img;
        warpPerspective(input_image, warped_img, M, Size(target_size, target_size));
        warped_images.push_back(warped_img);
    }
    return warped_images;
}


Mat CubeNumberRecognizer::jointCanvas(int target_size, vector<Mat> warped_images)
{
    int canvas_width = warped_images.size() * target_size;
    int canvas_height = target_size;
 
    Mat canvas(canvas_height, canvas_width, CV_8UC3, Scalar(0, 0, 0));

    for(int i = 0; i < warped_images.size(); i++)
    {
        //当前正视图的左上角坐标
        int x = i * target_size;
        int y = 0;
        //定义画布上的目标区域（ROI）
        Rect roi(x, y, target_size, target_size);
        //将当前正视图复制到ROI区域
        warped_images[i].copyTo(canvas(roi));
    }
    return canvas;
}


void CubeNumberRecognizer::visualizeResults(Mat& img, const vector<RectangleData>& rectangles) 
{
    vector<Scalar> colors = 
    {
        Scalar(255, 0, 0),    // 蓝色
        Scalar(0, 255, 0),    // 绿色
        Scalar(0, 0, 255),    // 红色
        Scalar(255, 255, 0),  // 青色
        Scalar(255, 0, 255),  // 紫色
        Scalar(0, 255, 255)   // 黄色
    };
    
    for (const auto& rect : rectangles) 
    {
        Scalar color = colors[(rect.id - 1) % colors.size()];
        
        // 1. 绘制方块边框
        for (size_t i = 0; i < rect.corners.size(); i++) 
        {
            Point2f p1 = rect.corners[i];
            Point2f p2 = rect.corners[(i + 1) % 4];
            line(img, p1, p2, color, 1);
        }
        
        // 2. 标记方块ID
        string id_text = "ID: " + to_string(rect.id);
        putText(img, id_text, rect.corners[0] + Point2f(-20, -10),
                FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
        
        // 3. 标记角点编号
        for (int i = 0; i < rect.corners.size(); i++) 
        {
            Point2f corner = rect.corners[i];
            // 绘制角点（实心圆）
            circle(img, corner, 2, Scalar(0, 255, 255), -1);
            // 标注角点序号
            string corner_text = to_string(i + 1);
            putText(img, corner_text, corner + Point2f(8, -5),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1);
        }
    }
}


void dockerMode_callback(const sensor_msgs::ImageConstPtr& msg)
{
    CubeNumberRecognizer recognizer; // 创建类实例
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch(const std::exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    Mat img_original = cv_ptr->image;
    recognizer.processImage(img_original);
    vector<RectangleData> rect_contours = recognizer.getCornerPoints(); 
    vector<Mat> warped_images = recognizer.perspecTransform(rect_contours, 200, img_original); 
    Mat canva = recognizer.jointCanvas(200, warped_images); 
    vector<int> number = recognizer.getDectedNumber(warped_images, num_templates, 200);

    
    for(const auto& rect : rect_contours)
    {
        ROS_INFO("\n\n==========Rectangles_ID:%d==========", rect.id);
        if (rect.id - 1 < number.size())
            ROS_INFO(" the number is %d", number[rect.id - 1]);
        else
            ROS_WARN("No number for Rectangles_ID:%d", rect.id);
        
        for(int i = 0; i < rect.corners.size(); i++)
        {
            ROS_INFO("Point_%d:(%f,%f)   ", i + 1, rect.corners[i].x, rect.corners[i].y);
        }
    }
    return;
}


void dockerMode()
{
    CubeNumberRecognizer recognizer; 
    num_templates = recognizer.loadNumberTemplates(temp_num, DOCKER_TEMPLATES_DIR, 200); 
    ROS_INFO("==========DOCKER_MODE==========");
    ros::NodeHandle nh;
    ros::Subscriber sub = nh.subscribe("/camera/color/image_raw", 1, dockerMode_callback);
    ros::spin();
}

// 本地模式：加载模板并处理单张图像
void localMode(const char *img_path)
{
    CubeNumberRecognizer recognizer; 
    num_templates = recognizer.loadNumberTemplates(temp_num, LOCAL_TEMPLATES_DIR, 200); 
    Mat img_original = imread(img_path, IMREAD_COLOR);
    if(img_original.empty())
    {
        ROS_WARN("can't find local images!!!");
        return; // 图像为空时退出，避免后续崩溃
    }
    recognizer.processImage(img_original); // 调用类成员函数
    vector<RectangleData> rect_contours = recognizer.getCornerPoints(); // 调用类成员函数
    vector<Mat> warped_images = recognizer.perspecTransform(rect_contours, 200, img_original); // 调用类成员函数
    Mat canva = recognizer.jointCanvas(200, warped_images); // 调用类成员函数
    vector<int> number = recognizer.getDectedNumber(warped_images, num_templates, 200); // 调用类成员函数

    // 输出检测结果（添加越界保护）
    for(const auto& rect : rect_contours)
    {
        ROS_INFO("\n\n==========Rectangles_ID:%d==========", rect.id);
        if (rect.id - 1 < number.size())
            ROS_INFO(" the number is %d", number[rect.id - 1]);
        else
            ROS_WARN("No number for Rectangles_ID:%d", rect.id);
        
        for(int i = 0; i < rect.corners.size(); i++)
        {
            ROS_INFO("Point_%d:(%f,%f)   ", i + 1, rect.corners[i].x, rect.corners[i].y);
        }
    }

    // 可视化
    Mat result = img_original.clone();
    recognizer.visualizeResults(result, rect_contours);
    namedWindow("检测结果", WINDOW_FREERATIO);
    imshow("检测结果", result);
    namedWindow("img_mask", WINDOW_FREERATIO);
    imshow("img_mask", red_mask);
    namedWindow("透视结果", WINDOW_FREERATIO);
    imshow("透视结果", canva);
    waitKey(0);
    destroyAllWindows(); 
}

int main(int argc, char  *argv[])
{
    setlocale(LC_ALL, ""); 
    ros::init(argc, argv, "cv_number_detect");
    
    if(argc == 1)
    {    
        dockerMode(); 
    }
    else if(argc == 2)
    {
        localMode(argv[1]); 
    }
    else
    {
        fprintf(stderr, "mistake! right way:\n");
        fprintf(stderr, "local image mode input:   rosrun cv_pkg cv_number_detect <imagepath>\n");
        fprintf(stderr, "docker mode input:   rosrun cv_pkg cv_number_detect\n"); // 修正：deocker → docker
        return -1;
    }
    return 0;
}
