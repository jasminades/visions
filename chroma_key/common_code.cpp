#include <iostream>
#include "common_code.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

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
fsiv_combine_images(const cv::Mat &img1, const cv::Mat &img2,
                    const cv::Mat &mask)
{
    CV_Assert(img2.size() == img1.size());
    CV_Assert(img2.type() == img1.type());
    CV_Assert(mask.size() == img1.size());
    cv::Mat output = img1.clone(); 
  

    img2.copyTo(output, mask);

    
    CV_Assert(output.size() == img1.size());
    CV_Assert(output.type() == img1.type());
    return output;
}

cv::Mat
fsiv_create_mask_from_hsv_range(const cv::Mat &hsv_img,
                                const cv::Scalar &lower_bound,
                                const cv::Scalar &upper_bound)
{
    CV_Assert(hsv_img.channels() == 3);
    cv::Mat mask;




    cv::inRange(hsv_img, lower_bound, upper_bound, mask);

    
    CV_Assert(mask.size() == hsv_img.size());
    CV_Assert(mask.depth() == CV_8U);
    return mask;
}

cv::Mat
fsiv_apply_chroma_key(const cv::Mat &foreg, const cv::Mat &backg, int hue,
                      int sensitivity)
{
    cv::Mat out;
    cv::Scalar lower_b, upper_b; 





    cv::Mat resized_background;
    cv::resize(backg, resized_background, foreg.size());


   
    int upper_hue = hue + sensitivity;
    int lower_hue = hue - sensitivity;

    cv::Scalar upper_bound(upper_hue, 255, 255);
    cv::Scalar lower_bound(lower_hue, 100, 100);



    cv::Mat hsv_foreground = fsiv_convert_bgr_to_hsv(foreg);
    cv::Mat mask = fsiv_create_mask_from_hsv_range(hsv_foreground, lower_bound, upper_bound);


 
    cv::Mat inv_mask;
    cv::bitwise_not(mask, inv_mask);



    cv::Mat foreground = fsiv_combine_images(foreg, foreg, inv_mask);
    cv::Mat background = fsiv_combine_images(resized_background, resized_background, mask );

    out = foreground + background;

    

    CV_Assert(out.size() == foreg.size());
    CV_Assert(out.type() == foreg.type());
    return out;
}
