#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "common_code.hpp"


int main(int argc, char *argv[]){
    if(argc != 6){
        std::cerr<< argv[0]<< " <input_image> <contrast> <brightness> <gamma> <luma>\n";
        return -1;
    }


    std::string input_image = argv[1];
    double contrast = 0.0;
    double gamma = 0.0;
    double brightness = 0.0;
    bool luma = false; 

    try{
        contrast = std::stod(argv[2]);
        gamma = std::stod(argv[3]);
        brightness = std::stod(argv[4]);
        luma = std::stoi(argv[5]);

    }catch(const std::invalid_argument& e){
        std::cerr<<"error";
        return -1;
    }

    cv::Mat image = cv::imread(input_image);

    if(image.empty()){
        std::cerr<<"err";
        return -1;
    }

    cv::Mat cbg_process = fsiv_cbg_process(image, contrast, brightness, gamma, luma);

    cv::imshow("original image ", image);
    cv::imshow("processed image ", cbg_process);
    cv::imwrite("cbg_process_image.jpg", cbg_process);
    cv::waitKey(0);

    return 0;
}