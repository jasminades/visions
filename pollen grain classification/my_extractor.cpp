
#include <opencv2/imgproc.hpp>
#include "my_extractor.hpp"
#include "gray_levels_features.hpp"

std::string MyExtractor::get_extractor_name() const
{
    return "my extractor";
}


MyExtractor::MyExtractor()
{
    type_ = FSIV_MY_EXTRACTOR;
    params_ = {0.0}; 
}

MyExtractor::~MyExtractor() {}

cv::Mat
MyExtractor::extract_features(const cv::Mat& img)
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
