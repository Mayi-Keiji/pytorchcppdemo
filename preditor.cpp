#include "preditor.h"
#include <map>
#include <iostream>
Preditor::Preditor(const std::string& model_path, bool use_gpu) :mDevice(torch::kCPU){
    if (torch::cuda::is_available() && use_gpu) {
        std::cout << "---use cuda---\n";
        mDevice = torch::kCUDA;
    } else {
        std::cout << "---use cpu---\n";
        mDevice = torch::kCPU;
    }

    try {
        mModule = torch::jit::load(model_path);
    }
    catch (const c10::Error& e) {
        std::cerr << "Loading the model failed!\n";
        std::exit(EXIT_FAILURE);
    }

    mHalf = (mDevice != torch::kCPU);
    mModule.to(mDevice);

    if (mHalf) {
        mModule.to(torch::kHalf);
    }
    mModule.eval();

    // init dict
    //{'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    mDict.insert(std::pair<int, std::string>(0, "daisy"));
    mDict.insert(std::pair<int, std::string>(1, "dandelion"));
    mDict.insert(std::pair<int, std::string>(2, "roses"));
    mDict.insert(std::pair<int, std::string>(3, "sunflower"));
    mDict.insert(std::pair<int, std::string>(4, "tulips"));
}

void Preditor::transforms(cv::Mat &input, torch::Tensor &out) {
    cv::resize(input, input, mSize);
    // 归一化
    input.convertTo(input, CV_32FC3, 1.0/255.0);
    auto tensor_input = torch::from_blob(input.data, { input.rows, input.cols, input.channels() }, at::kFloat);
    // [h,w,c] - > [c,h,w]
    tensor_input = tensor_input.permute({ 2,0,1 });
    // Normalize
    tensor_input[0] = tensor_input[0].sub_(0.5).div_(0.5);
    tensor_input[1] = tensor_input[1].sub_(0.5).div_(0.5);
    tensor_input[2] = tensor_input[2].sub_(0.5).div_(0.5);
    // 增加batch 通道
    tensor_input.unsqueeze_(0);
    out = tensor_input.to(mDevice);
}

bool Preditor::predit(cv::Mat &image, std::string &label) {
    torch::Tensor tensor;
    transforms(image, tensor);
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(tensor);
    auto output = mModule.forward(inputs).toTensor();
    std::cout << "output:" << output.softmax(1) << std::endl;
//    torch.softmax(output, dim=0)
    auto prediction_index = output.softmax(1).argmax().item<int>();
    std::cout << mDict[prediction_index] << std::endl;
    label = mDict[prediction_index];
    return true;
}
