#include <torch/script.h> // One-stop header.
#include <iostream>
#include <memory>
#include <sys/stat.h>
#include <fstream>
#include <string>
#include "utils.h"
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include "preditor.h"
using namespace std;




int predict_offical() {
    cout << "Hello World!" << endl;

    torch::jit::script::Module module;
    std::string path("D:\\AI\\Learn\\my_image_vision\\classification\\2_alexnet\\traced_alex_model.pt");
    std::cout << "file exists:" << file_exists(path) << std::endl;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(path);
    }
    catch (const c10::Error& e) {
        std::cout << "error loading the model\n";
        return -1;
    }
    std::cout << " loaded the model\n";
    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({1, 3, 224, 224}));

    // Execute the model and turn its output into a tensor.
    at::Tensor output = module.forward(inputs).toTensor();
    std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';


    std::cout << "ok\n";
    return 0;
}

void efficientb2_trace() {
    cv::Mat img = cv::imread("D:\\AI\\Learn\\engineer\\sunflower.jpg");
    cv::Mat imgCopy = img.clone();
    std::shared_ptr<Preditor> predit = std::make_shared<Preditor>("D:\\AI\\Learn\\my_image_vision\\classification\\10_efficientnetV2\\traced_efb2_model_traced.pt", true);
    std::string label;
    predit->predit(img,label);
    cv::putText(imgCopy, label, cv::Point(100,100), cv::FONT_HERSHEY_COMPLEX,1,cv::Scalar(0,0,255));
    cv::imshow("traced", imgCopy);
}

void efficientb2_script() {
    cv::Mat img = cv::imread("D:\\AI\\Learn\\engineer\\sunflower.jpg");
    cv::Mat imgCopy = img.clone();
    std::shared_ptr<Preditor> predit = std::make_shared<Preditor>("D:\\AI\\Learn\\my_image_vision\\classification\\10_efficientnetV2\\traced_efb2_model_scripted.pt", true);
    std::string label;
    predit->predit(img,label);
    cv::putText(imgCopy, label, cv::Point(100,100),cv::FONT_HERSHEY_COMPLEX,1,cv::Scalar(0,0,255));
    cv::imshow("script", imgCopy);
}

int main() {
    predict_offical();
    efficientb2_trace();
    efficientb2_script();
    cv::waitKey(0);
}
