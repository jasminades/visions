#include <iostream>
#include "common_code.hpp"
#include <opencv2/imgproc.hpp>

cv::Mat
fsiv_create_gaussian_filter(const int r)
{
    CV_Assert(r > 0);
    cv::Mat ret_v;
    //! TODO
    // Hint: use cv::getGaussianKernel()

    cv::Mat kernel = cv::getGaussianKernel(2 * r + 1, -1, CV_32F);
    ret_v = kernel * kernel.t();

    //
    CV_Assert(ret_v.type() == CV_32FC1);
    CV_Assert(ret_v.rows == (2 * r + 1) && ret_v.rows == ret_v.cols);
    CV_Assert(std::abs(1.0 - cv::sum(ret_v)[0]) < 1.0e-6);
    return ret_v;
}

cv::Mat fsiv_create_lap4_filter()
{
    cv::Mat ret_v;
    //! TODO
    // Hint: you can use a comma-separated initializer.
    //       See: https://docs.opencv.org/4.5.5/d6/d9e/classcv_1_1MatCommaInitializer__.html

    ret_v = (cv::Mat_<float>(3, 3) << 0, 1, 0,
                                    1, -4, 1,
                                    0, 1, 0);
    //
    CV_Assert(!ret_v.empty());
    CV_Assert(ret_v.rows == 3 && ret_v.cols == 3);
    CV_Assert(ret_v.type() == CV_32FC1);
    return ret_v;
}

cv::Mat fsiv_create_lap8_filter()
{
    cv::Mat ret_v;
    //! TODO
    // Hint: you can use a comma-separated initializer.
    //       See: https://docs.opencv.org/4.5.5/d6/d9e/classcv_1_1MatCommaInitializer__.html

    ret_v = (cv::Mat_<float>(3,3) << 1, 1, 1,
                                    1, -8, 1,
                                    1, 1, 1);
    //
    CV_Assert(!ret_v.empty());
    CV_Assert(ret_v.rows == 3 && ret_v.cols == 3);
    CV_Assert(ret_v.type() == CV_32FC1);
    return ret_v;
}

cv::Mat
fsiv_fill_expansion(cv::Mat const &in, const int r)
{
    CV_Assert(!in.empty());
    CV_Assert(r > 0);
    cv::Mat ret_v;
    //! TODO:
    // Hint: use cv::copyMakeBorder() using the constant value 0 to fill the
    //       expanded area.

    cv::copyMakeBorder(in, ret_v, r, r, r, r, cv::BORDER_CONSTANT, cv::Scalar(0));


    //
    CV_Assert(ret_v.type() == in.type());
    CV_Assert(ret_v.rows == in.rows + 2 * r);
    CV_Assert(ret_v.cols == in.cols + 2 * r);
    return ret_v;
}

cv::Mat
fsiv_circular_expansion(cv::Mat const &in, const int r)
{
    CV_Assert(!in.empty());
    CV_Assert(r > 0);
    cv::Mat ret_v;
    //! TODO
    //  Hint: use cv::copyMakeBorder() filling with border wrap extrapolation.

    cv::copyMakeBorder(in, ret_v, r, r, r, r, cv::BORDER_WRAP);

    //
    CV_Assert(ret_v.type() == in.type());
    CV_Assert(ret_v.rows == in.rows + 2 * r);
    CV_Assert(ret_v.cols == in.cols + 2 * r);
    CV_Assert(!(in.type() == CV_8UC1) || ret_v.at<uchar>(0, 0) == in.at<uchar>(in.rows - r, in.cols - r));
    CV_Assert(!(in.type() == CV_8UC1) || ret_v.at<uchar>(0, ret_v.cols / 2) == in.at<uchar>(in.rows - r, in.cols / 2));
    CV_Assert(!(in.type() == CV_8UC1) || ret_v.at<uchar>(0, ret_v.cols - 1) == in.at<uchar>(in.rows - r, r - 1));
    CV_Assert(!(in.type() == CV_8UC1) || ret_v.at<uchar>(ret_v.rows / 2, 0) == in.at<uchar>(in.rows / 2, in.cols - r));
    CV_Assert(!(in.type() == CV_8UC1) || ret_v.at<uchar>(ret_v.rows / 2, ret_v.cols / 2) == in.at<uchar>(in.rows / 2, in.cols / 2));
    CV_Assert(!(in.type() == CV_8UC1) || ret_v.at<uchar>(ret_v.rows - 1, 0) == in.at<uchar>(r - 1, in.cols - r));
    CV_Assert(!(in.type() == CV_8UC1) || ret_v.at<uchar>(ret_v.rows - 1, ret_v.cols / 2) == in.at<uchar>(r - 1, in.cols / 2));
    CV_Assert(!(in.type() == CV_8UC1) || ret_v.at<uchar>(ret_v.rows - 1, ret_v.cols - 1) == in.at<uchar>(r - 1, r - 1));
    return ret_v;
}

cv::Mat fsiv_create_dog_filter(int r1, int r2)
{
    CV_Assert(r1 > 0 && r1 < r2);
    cv::Mat ret_v;

    //! TODO
    // Hint: use fsiv_create_gaussian_filter() and fsiv_fill_expansion() to
    //       expand the smaller filter to be the same size as the larger one,
    //       so you can get the difference of both.

    cv::Mat gauss1 = fsiv_create_gaussian_filter(r1);
    cv::Mat gauss2 = fsiv_create_gaussian_filter(r2);

    gauss1 = fsiv_fill_expansion(gauss1, r2 - r1);

    ret_v = gauss2 - gauss1;
    //

    CV_Assert(!ret_v.empty());
    CV_Assert(ret_v.rows == (2 * r2 + 1) && ret_v.cols == (2 * r2 + 1));
    CV_Assert(ret_v.type() == CV_32FC1);
    return ret_v;
}

cv::Mat
fsiv_create_sharpening_filter(const int filter_type, int r1, int r2)
{
    CV_Assert(0 <= filter_type && filter_type <= 2);
    CV_Assert(filter_type != 2 || (0 < r1 && r1 < r2));
    cv::Mat filter;
    //! TODO
    //  Remember: sharpening filter = -Laplacian + [1]

    if(filter_type == 0){
        filter = fsiv_create_lap4_filter();
    }
    else if(filter_type == 1){
        filter = fsiv_create_lap8_filter();
    }
    else if(filter_type == 2){
        filter = fsiv_create_dog_filter(r1, r2);
    }

    cv::Mat identity = cv::Mat::zeros(filter.size(), CV_32FC1);
    identity.at<float>(filter.rows / 2, filter.cols / 2) = 1.0;
    filter = identity - filter;


    //
    CV_Assert(!filter.empty() && filter.type() == CV_32FC1);
    CV_Assert((filter_type == 2) || (filter.rows == 3 && filter.cols == 3));
    CV_Assert((filter_type != 2) || (filter.rows == (2 * r2 + 1) &&
                                     filter.cols == (2 * r2 + 1)));
    return filter;
}

cv::Mat
fsiv_image_sharpening(const cv::Mat &in, int filter_type,
                      int r1, int r2, bool circular)
{
    CV_Assert(in.type() == CV_32FC1);
    CV_Assert(0 < r1 && r1 < r2);
    CV_Assert(0 <= filter_type && filter_type <= 2);
    cv::Mat out;

    //! TODO
    // Remember: The effect consists of performing a convolution of the input
    //           image with the appropriate sharpening filter.
    // Hint: use cv::filter2D to convolve. In this case, because the input
    //       image is already extended, you should use 'cv::BORDER_ISOLATED'
    //       to do nothing to the extended area.
    // Remember: the convolved output image has the extended size but we need
    //           to return one with size equal to the input size. You can use
    //           cv::Mat::copyTo on a centered window to extract the result.

    cv::Mat filter = fsiv_create_sharpening_filter(filter_type, r1, r2);
    cv::Mat expanded_in = circular ? fsiv_circular_expansion(in, r2) : fsiv_fill_expansion(in, r2);
    
    cv::Mat convolved;
    cv::filter2D(expanded_in, convolved, -1, filter, cv::Point(-1, -1), 0, cv::BORDER_ISOLATED);

    out = convolved(cv::Rect(r2, r2, in.cols, in.rows)).clone();

    
    //
    CV_Assert(out.type() == in.type());
    CV_Assert(out.size() == in.size());
    return out;
}
