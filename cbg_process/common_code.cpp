#include "common_code.hpp"

cv::Mat
fsiv_convert_image_byte_to_float(const cv::Mat &img)
{
    CV_Assert(img.depth() == CV_8U);
    cv::Mat out;
   
    img.convertTo(out, CV_32F, 1.0 / 255.0);

    
    CV_Assert(out.rows == img.rows && out.cols == img.cols);
    CV_Assert(out.depth() == CV_32F);
    CV_Assert(img.channels() == out.channels());
    return out;
}

cv::Mat
fsiv_convert_image_float_to_byte(const cv::Mat &img)
{
    CV_Assert(img.depth() == CV_32F);
    cv::Mat out;
    
    img.convertTo(out, CV_8U, 255.0);

    
    CV_Assert(out.rows == img.rows && out.cols == img.cols);
    CV_Assert(out.depth() == CV_8U);
    CV_Assert(img.channels() == out.channels());
    return out;
}

cv::Mat
fsiv_convert_bgr_to_hsv(const cv::Mat &img)
{
    CV_Assert(img.channels() == 3);
    cv::Mat out;
    
    cv::cvtColor(img, out, cv::COLOR_BGR2HSV);

    
    CV_Assert(out.channels() == 3);
    return out;
}

cv::Mat
fsiv_convert_hsv_to_bgr(const cv::Mat &img)
{
    CV_Assert(img.channels() == 3);
    cv::Mat out;
    
    cv::cvtColor(img, out, cv::COLOR_HSV2BGR);

    
    CV_Assert(out.channels() == 3);
    return out;
}

cv::Mat
fsiv_cbg_process(const cv::Mat &in,
                 double contrast, double brightness, double gamma,
                 bool only_luma)
{
    CV_Assert(in.depth() == CV_8U);
    cv::Mat out;
    


    cv::Mat img_float;
    img_float = fsiv_convert_image_byte_to_float(in);


    if(only_luma && in.channels() == 3){
        cv::Mat hsv;

        cv::cvtColor(img_float, hsv, cv::COLOR_BGR2HSV);
        std::vector<cv::Mat> channels;

        cv::split(hsv, channels);


        
        cv::Mat luma;
        cv::pow(channels[2], gamma, luma);
        channels[2] = contrast * luma + brightness;

        cv::merge(channels, hsv);

        cv::cvtColor(hsv, img_float, cv::COLOR_HSV2BGR);
    }else{
        
        cv::Mat temp;
        cv::pow(img_float, gamma, temp);
        img_float = contrast * temp + brightness;
    }


    out = fsiv_convert_image_float_to_byte(img_float);
    


    
    CV_Assert(out.rows == in.rows && out.cols == in.cols);
    CV_Assert(out.depth() == CV_8U);
    CV_Assert(out.channels() == in.channels());
    return out;
}
