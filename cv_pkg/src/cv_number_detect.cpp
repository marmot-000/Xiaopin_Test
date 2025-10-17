#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>

using namespace cv;
using namespace std;
int temp_num = 27;  // 模板数量
int target_size = 200;
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
    vector<Mat> perspecTransform(const vector<RectangleData>& rect_contours,  Mat input_image); 
    vector<Mat> loadNumberTemplates(int temp_num, const string templates_dir);  // 加载数字模板
    vector<char> getDectedNumber(const vector<Mat> images_input, vector<Mat>& templates);  
    void visualizeResults(Mat& img, const vector<RectangleData>& rectangles);  // 可视化标注
    Mat jointCanvas(vector<Mat> warped_images); // 拼接多个方块透视的图
private:
    
};


vector<Mat> CubeNumberRecognizer::loadNumberTemplates(int temp_num, const string templates_dir)
{
    vector<Mat> num_templates;
    for(int i = 0; i < temp_num; i++)
    {
        string path = templates_dir + to_string(i + 1) + ".png";
        Mat temp = imread(path, IMREAD_GRAYSCALE);
        if (temp.empty())
        {
            ROS_ERROR("Temple file can not found: %s", path.c_str());
            return num_templates; 
        }
        resize(temp, temp, Size(target_size, target_size));
        num_templates.push_back(temp);
    }
    ROS_INFO("successfully loading templates, sum:%d", (int)num_templates.size());
    return num_templates;
}


vector<char> CubeNumberRecognizer::getDectedNumber(const vector<Mat> images_input, vector<Mat>& templates)
{
    vector<char> number;
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
        // 新增：输出最小差值和最佳匹配索引
        ROS_INFO("Min match value: %.2e, Best index: %d", min_val, best_match_idx);
        char detected_char = '\0'; // 用'\0'表示无效（避免-1导致乱码）
        if (min_val < 4.2e8 && best_match_idx != -1) 
        {
            if (best_match_idx >= 0 && best_match_idx <= 23) // 对应1-24.png（数字1-6）
            {
                int num = (best_match_idx + 1) / 4 + 1; // 4个模板对应1个数字
                detected_char = '0' + num; 
            }
            else if (best_match_idx == 24) 
            {
                detected_char = 'B';
            }
            else if (best_match_idx == 25) 
            {
                detected_char = 'O';
            }
            else if (best_match_idx == 26) 
            {
                detected_char = 'X';
            }
        }
        number.push_back(detected_char); 
    
    }
    return number;
    
}

void CubeNumberRecognizer::processImage(Mat img_original)
{
    //1.将RGB转换成HSV，便于截取
    Mat img_hsv;
    cvtColor(img_original, img_hsv, COLOR_BGR2HSV);
    //2.建两个掩膜
    inRange(img_hsv, Scalar(0, 112, 70), Scalar(10, 255, 255), mask_1);
    inRange(img_hsv, Scalar(160, 102, 70), Scalar(179, 255, 255), mask_2); 
    bitwise_or(mask_1, mask_2, red_mask);
    //3.开运算消除噪点
    Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(red_mask, red_mask, MORPH_OPEN, element);
    //4.闭运算连通孔洞
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
        if(area < 250) continue;
         // 形状接近正方形（宽高比在0.8~1.2之间,防止识别到robomaster E 的三个长方形）
        RotatedRect rect = minAreaRect(contour);
        float width = rect.size.width;
        float height = rect.size.height;
        float ratio = max(width, height) / min(width, height);
        if (ratio > 3) continue; 
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


vector<Mat> CubeNumberRecognizer::perspecTransform(const vector<RectangleData>& rect_contours, Mat input_image)
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


Mat CubeNumberRecognizer::jointCanvas(vector<Mat> warped_images)
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
    vector<Mat> warped_images = recognizer.perspecTransform(rect_contours, img_original); 
    Mat canva = recognizer.jointCanvas(warped_images); 
    vector<char> number = recognizer.getDectedNumber(warped_images, num_templates);

    
     for(const auto& rect : rect_contours)
    {
        ROS_INFO("\n\n==========Rectangles_ID:%d==========", rect.id);
        if (rect.id - 1 < number.size())
        {
            char det_char = number[rect.id - 1];
            if (det_char != '\0') // 有效结果
                ROS_INFO(" the number/char is %c", det_char); 
            else // 无效结果
                ROS_WARN("No valid number/char for Rectangles_ID:%d", rect.id);
        }
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
    num_templates = recognizer.loadNumberTemplates(temp_num, DOCKER_TEMPLATES_DIR); 
    ROS_INFO("==========DOCKER_MODE==========");
    ros::NodeHandle nh;
    ros::Subscriber sub = nh.subscribe("/camera/color/image_raw", 1, dockerMode_callback);
    ros::spin();
}

// 本地模式：加载模板并处理单张图像
void localMode(const char *img_path)
{
    CubeNumberRecognizer recognizer; 
    num_templates = recognizer.loadNumberTemplates(temp_num, LOCAL_TEMPLATES_DIR); 
    Mat img_original = imread(img_path, IMREAD_COLOR);
    if(img_original.empty())
    {
        ROS_WARN("can't find local images!!!");
        return; // 图像为空时退出，避免后续崩溃
    }
    recognizer.processImage(img_original);
    vector<RectangleData> rect_contours = recognizer.getCornerPoints(); 
    vector<Mat> warped_images = recognizer.perspecTransform(rect_contours, img_original); 
    Mat canva = recognizer.jointCanvas(warped_images);
    vector<char> number = recognizer.getDectedNumber(warped_images, num_templates); 

  
    for(const auto& rect : rect_contours)
    {
        ROS_INFO("\n\n==========Rectangles_ID:%d==========", rect.id);
        if (rect.id - 1 < number.size())
        {
            char det_char = number[rect.id - 1];
            if (det_char != '\0') // 有效结果
                ROS_INFO(" the number/char is %c", det_char); 
            else // 无效结果
                ROS_WARN("No valid number/char for Rectangles_ID:%d", rect.id);
        }
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
        fprintf(stderr, "docker mode input:   rosrun cv_pkg cv_number_detect\n"); 
        return -1;
    }
    return 0;
}

//下面这一段是有tracker可以调hsv
// #include <ros/ros.h>
// #include <cv_bridge/cv_bridge.h>
// #include <sensor_msgs/image_encodings.h>
// #include <opencv2/imgproc/imgproc.hpp>
// #include <opencv2/highgui/highgui.hpp>
// #include <opencv2/objdetect/objdetect.hpp>

// using namespace cv;
// using namespace std;

// // 全局参数：模板与尺寸配置
// int temp_num = 27;  // 模板数量
// int target_size = 200;  // 透视与模板统一尺寸
// const string DOCKER_TEMPLATES_DIR = "/opt/ep_ws/src/rmus_solution/temple/";
// const string LOCAL_TEMPLATES_DIR = "/home/robot/图片/temple/";
// vector<Mat> num_templates;  // 存储模板

// // HSV滑动条全局变量（初始值为红色常用范围）
// int h_low = 0, h_high = 10;
// int s_low = 100, s_high = 255;
// int v_low = 50, v_high = 255;
// int h_low2 = 160, h_high2 = 179;
// int s_low2 = 100, s_high2 = 255;
// int v_low2 = 50, v_high2 = 255;

// // 滑动条回调函数
// void onHSVTrackbar(int, void*) {}

// // 方块数据结构
// struct RectangleData 
// {
//     int id;  // 方块ID
//     vector<Point2f> corners;  // 4个角点（已排序）
// };

// class CubeNumberRecognizer
// {
// public:
//     void processImage(Mat img_original);  // 图像预处理
//     vector<RectangleData> getCornerPoints();  // 角点识别
//     vector<Mat> perspecTransform(const vector<RectangleData>& rect_contours, Mat input_image); 
//     vector<Mat> loadNumberTemplates(int temp_num, const string templates_dir); 
//     vector<char> getDectedNumber(const vector<Mat> images_input, vector<Mat>& templates);  
//     void visualizeResults(Mat& img, const vector<RectangleData>& rectangles);  
//     Mat jointCanvas(vector<Mat> warped_images); 
//     void initHSVTracker();  // 初始化HSV滑动条
//     Mat getRedMask() const { return red_mask; }  // 新增：公有函数获取red_mask（解决访问权限问题）
// private:
//     Mat mask_1, mask_2, red_mask;  // 私有成员，通过getRedMask()访问
// };

// // 1. 初始化HSV滑动条窗口
// void CubeNumberRecognizer::initHSVTracker()
// {
//     namedWindow("HSV 调节窗口", WINDOW_FREERATIO);
//     // 红色低区间滑动条
//     createTrackbar("H低(0-179)", "HSV 调节窗口", &h_low, 179, onHSVTrackbar);
//     createTrackbar("H高(0-179)", "HSV 调节窗口", &h_high, 179, onHSVTrackbar);
//     createTrackbar("S低(0-255)", "HSV 调节窗口", &s_low, 255, onHSVTrackbar);
//     createTrackbar("S高(0-255)", "HSV 调节窗口", &s_high, 255, onHSVTrackbar);
//     createTrackbar("V低(0-255)", "HSV 调节窗口", &v_low, 255, onHSVTrackbar);
//     createTrackbar("V高(0-255)", "HSV 调节窗口", &v_high, 255, onHSVTrackbar);
//     // 红色高区间滑动条
//     createTrackbar("H低2(0-179)", "HSV 调节窗口", &h_low2, 179, onHSVTrackbar);
//     createTrackbar("H高2(0-179)", "HSV 调节窗口", &h_high2, 179, onHSVTrackbar);
//     createTrackbar("S低2(0-255)", "HSV 调节窗口", &s_low2, 255, onHSVTrackbar);
//     createTrackbar("S高2(0-255)", "HSV 调节窗口", &s_high2, 255, onHSVTrackbar);
//     createTrackbar("V低2(0-255)", "HSV 调节窗口", &v_low2, 255, onHSVTrackbar);
//     createTrackbar("V高2(0-255)", "HSV 调节窗口", &v_high2, 255, onHSVTrackbar);
// }

// // 2. 图像预处理
// void CubeNumberRecognizer::processImage(Mat img_original)
// {
//     Mat img_hsv;
//     cvtColor(img_original, img_hsv, COLOR_BGR2HSV);
//     // 提取红色区域
//     inRange(img_hsv, Scalar(h_low, s_low, v_low), Scalar(h_high, s_high, v_high), mask_1);
//     inRange(img_hsv, Scalar(h_low2, s_low2, v_low2), Scalar(h_high2, s_high2, v_high2), mask_2);
//     bitwise_or(mask_1, mask_2, red_mask);
//     // 形态学操作
//     Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
//     morphologyEx(red_mask, red_mask, MORPH_OPEN, element);
//     morphologyEx(red_mask, red_mask, MORPH_CLOSE, element);
// }

// // 3. 加载模板
// vector<Mat> CubeNumberRecognizer::loadNumberTemplates(int temp_num, const string templates_dir)
// {
//     vector<Mat> num_templates;
//     for(int i = 0; i < temp_num; i++)
//     {
//         string path = templates_dir + to_string(i + 1) + ".png";
//         Mat temp = imread(path, IMREAD_GRAYSCALE);
//         if (temp.empty())
//         {
//             ROS_ERROR("模板文件未找到: %s", path.c_str());
//             return num_templates; 
//         }
//         resize(temp, temp, Size(target_size, target_size));
//         num_templates.push_back(temp);
//     }
//     ROS_INFO("成功加载模板数量: %d", (int)num_templates.size());
//     return num_templates;
// }

// // 4. 数字识别
// vector<char> CubeNumberRecognizer::getDectedNumber(const vector<Mat> images_input, vector<Mat>& templates)
// {
//     vector<char> number;
//     Mat digital_binary; 
//     for (auto& p : images_input) 
//     { 
//         double min_val = 1e9;
//         int best_match_idx = -1;
//         // 提取白字
//         Scalar lower_white = Scalar(85, 85, 95);
//         Scalar upper_white = Scalar(105, 105, 105);
//         inRange(p, lower_white, upper_white, digital_binary);
        
//         // 去噪+统一尺寸
//         Mat kernel = getStructuringElement(MORPH_RECT, Size(2, 2));
//         morphologyEx(digital_binary, digital_binary, MORPH_OPEN, kernel);
//         resize(digital_binary, digital_binary, Size(target_size, target_size));

//         // 模板匹配
//         for(int i = 0; i < templates.size(); i++)
//         {
//             Mat result;
//             matchTemplate(digital_binary, templates[i], result, TM_SQDIFF);
//             double val;
//             minMaxLoc(result, &val, nullptr, nullptr, nullptr);
//             if (val < min_val)
//             {
//                 min_val = val;
//                 best_match_idx = i;
//             } 
//         }

//         // 输出匹配信息
//         ROS_INFO("最小匹配值: %.2e, 最佳模板索引: %d", min_val, best_match_idx);
//         char detected_char = '\0';
//         if (min_val < 5e8 && best_match_idx != -1)
//         {
//             if (best_match_idx >= 0 && best_match_idx <= 23)
//             {
//                 int num = (best_match_idx + 1) / 4 + 1;
//                 detected_char = '0' + num;
//             }
//             else if (best_match_idx == 24) detected_char = 'B';
//             else if (best_match_idx == 25) detected_char = 'O';
//             else if (best_match_idx == 26) detected_char = 'X';
//         }
//         number.push_back(detected_char);
//     }
//     return number;
// }

// // 5. 角点识别（修复：Point→Point2f类型转换）
// vector<RectangleData> CubeNumberRecognizer::getCornerPoints()
// {
//     int current_id = 1;
//     vector<RectangleData> rect_contours;
//     vector<vector<Point>> contours;
//     Mat mask_copy = red_mask.clone();
//     findContours(mask_copy, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

//     for(auto &contour : contours)
//     {
//         // 过滤小轮廓
//         double area = contourArea(contour);
//         if(area < 250) continue;

//         // 过滤非正方形
//         RotatedRect rect = minAreaRect(contour);
//         float ratio = max(rect.size.width, rect.size.height) / min(rect.size.width, rect.size.height);
//         if (ratio > 1.2) continue;

//         // 多边形逼近（仅保留4角点）
//         double perimeter = arcLength(contour, true);
//         vector<Point> approx;
//         approxPolyDP(contour, approx, 0.02 * perimeter, true);
//         if(approx.size() != 4) continue;

//         // 角点排序：修复类型不匹配（Point→Point2f）
//         RectangleData rect_data;
//         rect_data.id = current_id++;
//         Point2f center(0, 0);  // 浮点型中心
//         for(auto& p : approx)
//         {
//             Point2f p_f(p.x, p.y);  // 整型Point→浮点型Point2f
//             rect_data.corners.push_back(p_f);
//             center += p_f;  // 同类型加法，编译通过
//         }
//         center /= 4;  // 计算中心坐标
//         // 按角度排序
//         sort(rect_data.corners.begin(), rect_data.corners.end(), 
//              [center](const Point2f& a, const Point2f& b){
//                  return atan2(a.y - center.y, a.x - center.x) < atan2(b.y - center.y, b.x - center.x);
//              });
//         rect_contours.push_back(rect_data);
//     }
//     return rect_contours;
// }

// // 6. 透视变换
// vector<Mat> CubeNumberRecognizer::perspecTransform(const vector<RectangleData>& rect_contours, Mat input_image)
// {
//     vector<Mat> warped_images;
//     vector<Point2f> dst = { Point2f(0,0), Point2f(target_size,0), Point2f(target_size,target_size), Point2f(0,target_size) };
//     for(auto& rect_contour : rect_contours)
//     {
//         Mat M = getPerspectiveTransform(rect_contour.corners, dst);
//         Mat warped_img;
//         warpPerspective(input_image, warped_img, M, Size(target_size, target_size));
//         warped_images.push_back(warped_img);
//     }
//     return warped_images;
// }

// // 7. 画布拼接
// Mat CubeNumberRecognizer::jointCanvas(vector<Mat> warped_images)
// {
//     if (warped_images.empty()) return Mat();
//     int canvas_width = warped_images.size() * target_size;
//     Mat canvas(Size(canvas_width, target_size), CV_8UC3, Scalar(0, 0, 0));
//     for(int i = 0; i < warped_images.size(); i++)
//     {
//         Rect roi(i * target_size, 0, target_size, target_size);
//         warped_images[i].copyTo(canvas(roi));
//     }
//     return canvas;
// }

// // 8. 可视化标注
// void CubeNumberRecognizer::visualizeResults(Mat& img, const vector<RectangleData>& rectangles) 
// {
//     vector<Scalar> colors = {Scalar(255,0,0), Scalar(0,255,0), Scalar(0,0,255), Scalar(255,255,0)};
//     for (const auto& rect : rectangles) 
//     {
//         Scalar color = colors[(rect.id - 1) % colors.size()];
//         // 绘制边框
//         for (size_t i = 0; i < 4; i++)
//             line(img, rect.corners[i], rect.corners[(i+1)%4], color, 2);
//         // 标注ID
//         string id_text = "ID: " + to_string(rect.id);
//         putText(img, id_text, rect.corners[0] + Point2f(-20, -10), FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
//         // 标注角点
//         for (int i = 0; i < 4; i++)
//         {
//             circle(img, rect.corners[i], 3, Scalar(0,255,255), -1);
//             putText(img, to_string(i+1), rect.corners[i] + Point2f(5, -5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,255), 1);
//         }
//     }
// }

// // 9. Docker模式回调（修复：用getRedMask()获取私有成员）
// void dockerMode_callback(const sensor_msgs::ImageConstPtr& msg)
// {
//     static CubeNumberRecognizer recognizer;  // 静态实例，避免重复创建
//     cv_bridge::CvImagePtr cv_ptr;
//     try
//     {
//         cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
//     }
//     catch(const std::exception& e)
//     {
//         ROS_ERROR("cv_bridge错误: %s", e.what());
//         return;
//     }
//     Mat img_original = cv_ptr->image;

//     // 预处理
//     recognizer.processImage(img_original);
//     vector<RectangleData> rect_contours = recognizer.getCornerPoints();
//     vector<Mat> warped_imgs = recognizer.perspecTransform(rect_contours, img_original);
//     vector<char> number = recognizer.getDectedNumber(warped_imgs, num_templates);

//     // 输出结果
//     for(const auto& rect : rect_contours)
//     {
//         ROS_INFO("\n========== 方块ID: %d ==========", rect.id);
//         if (rect.id - 1 < number.size())
//         {
//             char c = number[rect.id - 1];
//             if (c != '\0') ROS_INFO("识别结果: %c", c);
//             else ROS_WARN("无有效识别结果");
//         }
//         for(int i = 0; i < 4; i++)
//             ROS_INFO("角点%d: (%.2f, %.2f)", i+1, rect.corners[i].x, rect.corners[i].y);
//     }

//     // 实时显示：用getRedMask()获取red_mask（修复访问权限）
//     Mat result = img_original.clone();
//     recognizer.visualizeResults(result, rect_contours);
//     imshow("Docker模式-识别结果", result);
//     imshow("Docker模式-红色掩码", recognizer.getRedMask());  // 修复：调用公有函数
//     waitKey(1);
// }

// // 10. Docker模式初始化
// void dockerMode()
// {
//     CubeNumberRecognizer recognizer;
//     recognizer.initHSVTracker();
//     num_templates = recognizer.loadNumberTemplates(temp_num, DOCKER_TEMPLATES_DIR);
//     if (num_templates.empty())
//     {
//         ROS_ERROR("模板加载失败，退出Docker模式");
//         return;
//     }

//     ROS_INFO("========== Docker模式启动 ==========");
//     ros::NodeHandle nh;
//     ros::Subscriber sub = nh.subscribe("/camera/color/image_raw", 1, dockerMode_callback);
//     ros::spin();
//     destroyAllWindows();
// }

// // 11. 本地模式（修复：用getRedMask()获取私有成员）
// void localMode(const char *img_path)
// {
//     CubeNumberRecognizer recognizer;
//     recognizer.initHSVTracker();
//     num_templates = recognizer.loadNumberTemplates(temp_num, LOCAL_TEMPLATES_DIR);
//     if (num_templates.empty())
//     {
//         ROS_ERROR("模板加载失败，退出本地模式");
//         return;
//     }

//     // 读取本地图像
//     Mat img_original = imread(img_path, IMREAD_COLOR);
//     if(img_original.empty())
//     {
//         ROS_WARN("图像未找到: %s", img_path);
//         return;
//     }

//     // 循环调试
//     while (true)
//     {
//         // 预处理+识别
//         recognizer.processImage(img_original);
//         vector<RectangleData> rect_contours = recognizer.getCornerPoints();
//         vector<Mat> warped_imgs = recognizer.perspecTransform(rect_contours, img_original);
//         Mat canva = recognizer.jointCanvas(warped_imgs);
//         vector<char> number = recognizer.getDectedNumber(warped_imgs, num_templates);

//         // 输出结果
//         for(const auto& rect : rect_contours)
//         {
//             ROS_INFO("\n========== 方块ID: %d ==========", rect.id);
//             if (rect.id - 1 < number.size())
//             {
//                 char c = number[rect.id - 1];
//                 if (c != '\0') ROS_INFO("识别结果: %c", c);
//                 else ROS_WARN("无有效识别结果");
//             }
//             for(int i = 0; i < 4; i++)
//                 ROS_INFO("角点%d: (%.2f, %.2f)", i+1, rect.corners[i].x, rect.corners[i].y);
//         }

//         // 可视化：用getRedMask()获取red_mask（修复访问权限）
//         Mat result = img_original.clone();
//         recognizer.visualizeResults(result, rect_contours);
//         namedWindow("本地模式-识别结果", WINDOW_FREERATIO);
//         namedWindow("本地模式-红色掩码", WINDOW_FREERATIO);
//         namedWindow("本地模式-透视拼接", WINDOW_FREERATIO);
//         imshow("本地模式-识别结果", result);
//         imshow("本地模式-红色掩码", recognizer.getRedMask());  // 修复：调用公有函数
//         if (!canva.empty()) imshow("本地模式-透视拼接", canva);

//         // 按 'q' 退出
//         if (waitKey(30) == 'q') break;
//     }
//     destroyAllWindows();
// }

// // 主函数
// int main(int argc, char  *argv[])
// {
//     setlocale(LC_ALL, ""); 
//     ros::init(argc, argv, "cv_number_detect");
    
//     if(argc == 1) dockerMode();    // 无参数：Docker模式
//     else if(argc == 2) localMode(argv[1]);  // 有参数：本地模式
//     else
//     {
//         fprintf(stderr, "参数错误！正确用法：\n");
//         fprintf(stderr, "1. Docker模式: rosrun cv_pkg cv_number_detect\n");
//         fprintf(stderr, "2. 本地模式: rosrun cv_pkg cv_number_detect <图像路径>\n");
//         return -1;
//     }
//     return 0;
// }

