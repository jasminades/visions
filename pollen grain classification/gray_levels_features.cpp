
#include <opencv2/imgproc.hpp>
#include "gray_levels_features.hpp"

std::string
GrayLevelsFeatures::get_extractor_name () const
{
    std::string name = "Gray levels ";
    switch (int(params_[0]))
    {
        case 0:
            name += "[0,1] normalized.";
            break;
        case 1:
            name += "mean,stddev normalized.";
            break;                   

        default:
            throw std::runtime_error("unknown type of gray level extractor.");
            break;
    }
    return name;
}

cv::Mat
fsiv_extract_01_normalized_graylevels (cv::Mat const& img)
{
    cv::Mat feature;
    cv::Mat gray;
    
    if(img.channels() == 3){
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    }else{
        gray = img;
    }

    gray.convertTo(feature, CV_32F, 1.0 / 255.0);

    feature = feature.reshape(1,1);
    
    CV_Assert(feature.rows==1);
    CV_Assert(feature.type()==CV_32FC1);
    return feature;
}

cv::Mat
fsiv_extract_mean_stddev_normalized_gray_levels(cv::Mat const& img)
{
    
    cv::Mat feature;
    cv::Mat gray;
   
    if(img.channels() == 3){
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    }else{
        gray = img;
    }

    cv::Scalar mean, stddev;
    cv::meanStdDev(gray, mean, stddev);


    gray.convertTo(feature, CV_32F, 1.0, -mean[0]);
    feature = feature / (stddev[0] + 1e-6);

    feature = feature.reshape(1,1);

    CV_Assert(feature.rows==1);
    CV_Assert(feature.type()==CV_32FC1);
    return feature;
}

GrayLevelsFeatures::GrayLevelsFeatures()
{
    type_ = FSIV_GREY_LEVELS;
    params_ = {0.0};
}

GrayLevelsFeatures::~GrayLevelsFeatures() {}

cv::Mat
GrayLevelsFeatures::extract_features(const cv::Mat& img)
{    
    cv::Mat feature;
    switch (int(params_[0]))
    {
        case 0:
            feature = fsiv_extract_01_normalized_graylevels(img);
            break;
        case 1:
            feature = fsiv_extract_mean_stddev_normalized_gray_levels(img);
            break;
        default:
            throw std::runtime_error("Unknown gray level feature extractor type: " 
                + std::to_string(int(params_[0])));
            break;
    }
    CV_Assert(feature.rows==1);
    CV_Assert(feature.type()==CV_32FC1);
    return feature;
}
