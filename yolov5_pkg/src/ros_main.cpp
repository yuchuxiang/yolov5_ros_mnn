//
// Created by ycx on 22-7-7.
//

#include <iostream>
#include <string>
#include<ctime>

#include <MNN/MNNDefine.h>
#include <MNN/MNNForwardType.h>
#include <MNN/Interpreter.hpp>
#include <opencv2/opencv.hpp>
#include "yolov5_pkg/Yolo.h"

#include "ros/ros.h"
#include "sensor_msgs/Image.h"
#include "cv_bridge/cv_bridge.h"


class YOLO_ROS{
private:

    ros::NodeHandle nh;

    ros::Subscriber sub_cam;

    ros::Publisher  pub_box;

    cv::Mat image_mat;

    cv_bridge::CvImagePtr cv_ptr;

    sensor_msgs::ImagePtr cv_ptr_to_sensor;

    sensor_msgs::ImagePtr sensor_image;

    sensor_msgs::ImagePtr sensor_image_out;

    std_msgs::Header header;

public:
    YOLO_ROS(){//构造函数

        sub_cam = nh.subscribe("/usb_cam/image_raw",1,&YOLO_ROS::image_callback, this,ros::TransportHints().tcpNoDelay());

        pub_box = nh.advertise<sensor_msgs::Image>("/yolo_image",1);

        sensor_image.reset(new sensor_msgs::Image);

        cv_ptr.reset(new cv_bridge::CvImage);

        cv_ptr_to_sensor.reset(new sensor_msgs::Image);

}


    void image_callback(const sensor_msgs::ImageConstPtr & image_in){
        //这段代码要将读取到的视频流转换为单帧图片
        *sensor_image = *image_in;

        header = image_in->header;

        header.stamp = ros::Time::now(); // time

        cv_ptr = cv_bridge::toCvCopy(sensor_image,sensor_msgs::image_encodings::TYPE_8UC3);

        image_mat = cv_ptr->image;

        systeam(image_mat);
    }

    void systeam(cv::Mat mat_in){

        int num_classes=80;
        std::vector<YoloLayerData> yolov5s_layers{
                {"937",    32, {{146, 217}, {231, 300}, {335, 433}}},
                {"917",    16, {{23,  29}, {43,  55},  {73,  105}}},
                {"output", 8,  {{4,  5}, {8,  10},  {13,  16}}},
        };
        std::vector<YoloLayerData> & layers = yolov5s_layers;
        int net_size =640;
        std::string model_name = "/home/ycx/yolov5_ros/src/yolov5_pkg/model_zoo/v5lite-s.mnn";
        std::shared_ptr<MNN::Interpreter> net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_name.c_str()));
        if (nullptr == net) {
            return ;
        }

        MNN::ScheduleConfig config;
        config.numThread = 4;
        config.type      = static_cast<MNNForwardType>(MNN_FORWARD_CPU);
        MNN::BackendConfig backendConfig;
        backendConfig.precision = (MNN::BackendConfig::PrecisionMode)2;
        // backendConfig.precision =  MNN::PrecisionMode Precision_Normal; // static_cast<PrecisionMode>(Precision_Normal);
        config.backendConfig = &backendConfig;
        MNN::Session *session = net->createSession(config);

        int INPUT_SIZE = 640;
        // load image
        cv::Mat raw_image    = mat_in;
        cv::Mat image;
        cv::resize(raw_image, image, cv::Size(INPUT_SIZE, INPUT_SIZE));

        // preprocessing
        image.convertTo(image, CV_32FC3);
        // image = (image * 2 / 255.0f) - 1;
        image = image /255.0f;

        // wrapping input tensor, convert nhwc to nchw
        std::vector<int> dims{1, INPUT_SIZE, INPUT_SIZE, 3};
        auto nhwc_Tensor = MNN::Tensor::create<float>(dims, NULL, MNN::Tensor::TENSORFLOW);
        auto nhwc_data   = nhwc_Tensor->host<float>();
        auto nhwc_size   = nhwc_Tensor->size();
        std::memcpy(nhwc_data, image.data, nhwc_size);

        auto inputTensor = net->getSessionInput(session, nullptr);
        inputTensor->copyFromHostTensor(nhwc_Tensor);

        // run network
        clock_t startTime,endTime;
        startTime = clock();//计时开始
        net->runSession(session);
        endTime = clock();//计时结束
        cout << "The forward time is: " <<(double)(endTime - startTime) / 1000.0 << "ms" << endl;


        // get output data
        std::string output_tensor_name0 = layers[2].name ;
        std::string output_tensor_name1 = layers[1].name ;
        std::string output_tensor_name2 = layers[0].name ;



        MNN::Tensor *tensor_scores  = net->getSessionOutput(session, output_tensor_name0.c_str());
        MNN::Tensor *tensor_boxes   = net->getSessionOutput(session, output_tensor_name1.c_str());
        MNN::Tensor *tensor_anchors = net->getSessionOutput(session, output_tensor_name2.c_str());

        MNN::Tensor tensor_scores_host(tensor_scores, tensor_scores->getDimensionType());
        MNN::Tensor tensor_boxes_host(tensor_boxes, tensor_boxes->getDimensionType());
        MNN::Tensor tensor_anchors_host(tensor_anchors, tensor_anchors->getDimensionType());

        tensor_scores->copyToHostTensor(&tensor_scores_host);
        tensor_boxes->copyToHostTensor(&tensor_boxes_host);
        tensor_anchors->copyToHostTensor(&tensor_anchors_host);

        std::vector<BoxInfo> result;
        std::vector<BoxInfo> boxes;

        yolocv::YoloSize yolosize = yolocv::YoloSize{INPUT_SIZE,INPUT_SIZE};

        float threshold = 0.25;
        float nms_threshold = 0.5;

        // show_shape(tensor_scores_host.shape());
        // show_shape(tensor_boxes_host.shape());
        // show_shape(tensor_anchors_host.shape());


        boxes = decode_infer(tensor_scores_host, layers[2].stride,  yolosize, net_size, num_classes, layers[2].anchors, threshold);
        result.insert(result.begin(), boxes.begin(), boxes.end());

        boxes = decode_infer(tensor_boxes_host, layers[1].stride,  yolosize, net_size, num_classes, layers[1].anchors, threshold);
        result.insert(result.begin(), boxes.begin(), boxes.end());

        boxes = decode_infer(tensor_anchors_host, layers[0].stride,  yolosize, net_size, num_classes, layers[0].anchors, threshold);
        result.insert(result.begin(), boxes.begin(), boxes.end());

        nms(result, nms_threshold);

        scale_coords(result, INPUT_SIZE, INPUT_SIZE, raw_image.cols, raw_image.rows);
        cv::Mat frame_show = draw_box(raw_image, result);

        //将frame_show转换为sensor_msgs
        cv_ptr_to_sensor = cv_bridge::CvImage(header, "bgr8", frame_show).toImageMsg();

        sensor_image_out = cv_ptr_to_sensor;

        pub_box.publish(sensor_image_out);
        //cv::imwrite("output.jpg", frame_show);

    }

    void show_shape(std::vector<int> shape)
    {
        std::cout<<shape[0]<<" "<<shape[1]<<" "<<shape[2]<<" "<<shape[3]<<" "<<shape[4]<<" "<<std::endl;

    }

    void scale_coords(std::vector<BoxInfo> &boxes, int w_from, int h_from, int w_to, int h_to)
    {
        float w_ratio = float(w_to)/float(w_from);
        float h_ratio = float(h_to)/float(h_from);


        for(auto &box: boxes)
        {
            box.x1 *= w_ratio;
            box.x2 *= w_ratio;
            box.y1 *= h_ratio;
            box.y2 *= h_ratio;
        }
        return ;
    }

    cv::Mat draw_box(cv::Mat & cv_mat, std::vector<BoxInfo> &boxes)
    {
        int CNUM = 80;
        static const char* class_names[] = {
                "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
                "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
                "hair drier", "toothbrush"
        };
        cv::RNG rng(0xFFFFFFFF);
        cv::Scalar_<int> randColor[CNUM];
        for (int i = 0; i < CNUM; i++)
            rng.fill(randColor[i], cv::RNG::UNIFORM, 0, 256);

        for(auto box : boxes)
        {
            int width = box.x2-box.x1;
            int height = box.y2-box.y1;
            int id = box.id;
            char text[256];
            cv::Point p = cv::Point(box.x1, box.y1-5);
            cv::Rect rect = cv::Rect(box.x1, box.y1, width, height);
            cv::rectangle(cv_mat, rect, cv::Scalar(0, 0, 255));
            sprintf(text, "%s %.1f%%", class_names[box.label], box.score * 100);
            cv::putText(cv_mat, text, p, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
        }
        return cv_mat;
    }

};



int main(int argc, char **argv)
{
    ros::init(argc, argv, "yolov5");

    ROS_INFO("\033[1;32m----> YOLOV5 Started.\033[0m");

    YOLO_ROS yoloros;

    ros::spin();

    return 0;

}




// std::shared_ptr<MNN::Interpreter> net =
//     std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(fileName));
// if (nullptr == net) {
//     return 0;
// }
// // Must call it before createSession.
// // If ".tempcache" file does not exist, Interpreter will go through the
// // regular initialization procedure, after which the compiled model files
// // will be written  to ".tempcache".
// // If ".tempcache" file exists, the Interpreter will be created from the
// // cache.
// net->setCacheFile(".tempcache");

// MNN::ScheduleConfig config;
// // Creates the session after you've called setCacheFile.
// MNN::Session* session = net->createSession(config);

