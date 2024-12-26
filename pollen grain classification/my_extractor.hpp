/**
 *  @file my_extractor.hpp
 *  (C) 2022- FJMC fjmadrid@uco.es
 */
#pragma once

#include "features.hpp"
#include <opencv2/core.hpp>

class MyExtractor: public FeaturesExtractor
{
public:
    /**
     * @brief Create and set the default parameters.
     */
    MyExtractor();
    ~MyExtractor();

    virtual std::string get_extractor_name() const override;
    virtual cv::Mat extract_features(const cv::Mat& img) override;

    //This extractor does not need override these methods:
    //virtual void train(const cv::Mat& samples) override;
    //virtual bool save_model(std::string const& fname) const;
    //virtual bool load_model(std::string const& fname);

};
