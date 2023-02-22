#ifndef PREDICTOR_H
#define PREDICTOR_H
#include <torch/script.h> // One-stop header.
#include <string>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <map>
class Preditor
{
public:
    Preditor(const std::string& model_path, bool use_gpu);
    bool predit(cv::Mat &image, std::string &label);
    void transforms(cv::Mat &input, torch::Tensor &out);
private:
    torch::jit::script::Module mModule;
    torch::Device mDevice;
    bool mHalf;
    cv::Size mSize = cv::Size(224,224);
    std::map<int, std::string> mDict;
};

#endif // PREDICTOR_H
