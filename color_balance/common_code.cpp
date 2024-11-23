#include "common_code.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

cv::Mat fsiv_color_rescaling(const cv::Mat &in, const cv::Scalar &from, const cv::Scalar &to)
{
    CV_Assert(in.type() == CV_8UC3);
    cv::Mat out;
    cv::Scalar scale;
    cv::divide(to, from, scale);
    out = in.mul(scale);
    CV_Assert(out.type() == in.type());
    CV_Assert(out.rows == in.rows && out.cols == in.cols);
    return out;
}

cv::Mat fsiv_gray_world_color_balance(cv::Mat const &in)
{
    CV_Assert(in.type() == CV_8UC3);
    cv::Mat out;
    cv::Scalar meanColor = cv::mean(in);
    cv::Scalar scale(128.0 / meanColor[0], 128.0 / meanColor[1], 128.0 / meanColor[2]);
    out = in.mul(scale);
    CV_Assert(out.type() == in.type());
    CV_Assert(out.rows == in.rows && out.cols == in.cols);
    return out;
}

cv::Mat fsiv_convert_bgr_to_gray(const cv::Mat &img, cv::Mat &out)
{
    CV_Assert(img.channels() == 3);
    cv::cvtColor(img, out, cv::COLOR_BGR2GRAY);
    CV_Assert(out.channels() == 1);
    return out;
}

cv::Mat fsiv_compute_image_histogram(cv::Mat const &img)
{
    CV_Assert(img.type() == CV_8UC1);
    cv::Mat hist;
    int histogram_size = 256;
    float range[] = {0, 256};
    const float* histogram_range = {range};

    cv::calcHist(&img, 1, 0, cv::Mat(), hist, 1, &histogram_size, &histogram_range);
    cv::normalize(hist, hist, 1, 0, cv::NORM_L1);

    CV_Assert(!hist.empty());
    CV_Assert(hist.type() == CV_32FC1);
    CV_Assert(hist.rows == 256 && hist.cols == 1);
    return hist;
}

float fsiv_compute_histogram_percentile(cv::Mat const &hist, float p_value)
{
    CV_Assert(hist.type() == CV_32FC1);
    CV_Assert(hist.cols == 1);
    CV_Assert(0.0 <= p_value && p_value <= 1.0);

    int p = 0;
    float cumulative = 0.0;
    float target = p_value * cv::sum(hist)[0];

    for(int i = 0; i < hist.rows; ++i){
        cumulative += hist.at<float>(i);
        if(cumulative >= target){
            p = i;
            break;
        }
    }

    CV_Assert(0 <= p && p < hist.rows);
    return p;
}

cv::Mat fsiv_white_patch_color_balance(cv::Mat const &in, float p)
{
    CV_Assert(in.type() == CV_8UC3);
    CV_Assert(0.0f <= p && p <= 100.0f);
    cv::Mat out;

    if (p == 0.0)
    {
        cv::Mat gray;
        cv::cvtColor(in, gray, cv::COLOR_BGR2GRAY);

        double minimalValue;
        double maximumValue;
        cv::Point minLoc;
        cv::Point maxLoc;
        cv::minMaxLoc(gray, &minimalValue, &maximumValue, &minLoc, &maxLoc);

        cv::Vec3b brightest = in.at<cv::Vec3b>(maxLoc);
        cv::Scalar from(brightest[2], brightest[1], brightest[0]);
        out = fsiv_color_rescaling(in, from, cv::Scalar(255, 255, 255));
    }
    else
    {
        cv::Mat gray;
        cv::cvtColor(in, gray, cv::COLOR_BGR2GRAY);

        cv::Mat hist = fsiv_compute_image_histogram(gray);
        float percentile = fsiv_compute_histogram_percentile(hist, (100 - p) / 100.0);

        cv::Mat mask = gray >= percentile;
        cv::Scalar meanValue = cv::mean(in, mask);

        out = fsiv_color_rescaling(in, meanValue, cv::Scalar(255, 255, 255));
    }

    CV_Assert(out.type() == in.type());
    CV_Assert(out.rows == in.rows && out.cols == in.cols);
    return out;
}
