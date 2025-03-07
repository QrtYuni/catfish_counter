#ifndef INFERENCE_H
#define INFERENCE_H

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <random>
#include <iostream>

using namespace std;

struct Detection
{
    int class_id{0};
    string className{};
    float confidence{0.0};
    cv::Scalar color{};
    cv::Rect box{};
};

class Inference
{
public:
    Inference(const string &onnxModelPath, const cv::Size &modelInputShape = {640,640}, const string &classesTxtFile = "", const bool &runWithCuda = true);
    vector<Detection> runInference(const cv::Mat &input);

private:
    void loadOnnxNetwork();
    cv::Mat formatToSquare(const cv::Mat &source);

    string modelPath{};
    string classesPath{};
    bool cudaEnable{};

    vector<string> classes{"b", "gb", "gk"};

    cv::Size2f modelShape{};

    float modelConfidenceThreshold  {0.25};
    float modelScoreThreshold       {0.25};
    float modelNMSThreshold         {0.25};     

    bool letterBoxForSquare = true;
    cv::dnn::Net net;

};

#endif