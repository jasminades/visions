
#pragma once

#include "features.hpp"

class GrayLevelsFeatures: public FeaturesExtractor
{
public:
    /**
     * @brief Create and set the default parameters.
     */
    GrayLevelsFeatures();
    ~GrayLevelsFeatures();

    virtual std::string get_extractor_name() const override;
    virtual cv::Mat extract_features(const cv::Mat& img) override;


};

cv::Mat fsiv_extract_01_normalized_graylevels(const cv::Mat& img);

cv::Mat fsiv_extract_mean_stddev_normalized_gray_levels(const cv::Mat& img);
