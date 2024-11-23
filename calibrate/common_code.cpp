#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include "common_code.hpp"

std::vector<cv::Point3f>
fsiv_generate_3d_calibration_points(const cv::Size &board_size,
                                    float square_size)
{
    std::vector<cv::Point3f> ret_v;
   

    for(int i = 0; i < board_size.height; ++i)
        for(int j = 0; j < board_size.width; ++j)
            ret_v.emplace_back((j + 1) * square_size, (i + 1) * square_size, 0.0f);

    
    CV_Assert(ret_v.size() == static_cast<size_t>(board_size.width * board_size.height));
    return ret_v;
}

bool fsiv_find_chessboard_corners(const cv::Mat &img, const cv::Size &board_size,
                                  std::vector<cv::Point2f> &corner_points,
                                  const char *wname)
{
    CV_Assert(img.type() == CV_8UC3);
    bool was_found = false;
    
    was_found = cv::findChessboardCorners(img, board_size, corner_points);

    if(was_found){
        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
       
        cv::cornerSubPix(gray, corner_points, cv::Size(5, 5), cv::Size(-1, -1), 
                 cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 40, 0.01));


        if(wname != nullptr){
            cv::drawChessboardCorners(img, board_size, corner_points, was_found);
            cv::imshow(wname, img);
            if(cv::waitKey(0) == 27)
                return false;
        }
    }
    
    return was_found;
}

float fsiv_calibrate_camera(const std::vector<std::vector<cv::Point2f>> &_2d_points,
                            const std::vector<std::vector<cv::Point3f>> &_3d_points,
                            const cv::Size &camera_size,
                            cv::Mat &camera_matrix,
                            cv::Mat &dist_coeffs,
                            std::vector<cv::Mat> *rvecs,
                            std::vector<cv::Mat> *tvecs)
{
    CV_Assert(_3d_points.size() >= 2 && _3d_points.size() == _2d_points.size());
    float error = 0.0;
    
    camera_matrix = (cv::Mat_<double>(3, 3) << 800, 0, 320, 0, 800, 240, 0, 0, 1);
    dist_coeffs = (cv::Mat_<double>(1, 5) << -0.5, 0.2, 0, 0, 0);


    error = cv::calibrateCamera(_3d_points, _2d_points, camera_size, camera_matrix, dist_coeffs, 
                                rvecs != nullptr ? *rvecs : std::vector<cv::Mat>(),
                                tvecs != nullptr ? *tvecs : std::vector<cv::Mat>());
  
    CV_Assert(camera_matrix.rows == camera_matrix.cols &&
              camera_matrix.rows == 3 &&
              camera_matrix.type() == CV_64FC1);
    CV_Assert((dist_coeffs.rows * dist_coeffs.cols) == 5 &&
              dist_coeffs.type() == CV_64FC1);
    CV_Assert(rvecs == nullptr || rvecs->size() == _2d_points.size());
    CV_Assert(tvecs == nullptr || tvecs->size() == _2d_points.size());
    return error;
}

void fsiv_save_calibration_parameters(cv::FileStorage &fs,
                                      const cv::Size &camera_size,
                                      float error,
                                      const cv::Mat &camera_matrix,
                                      const cv::Mat &dist_coeffs,
                                      const cv::Mat &rvec,
                                      const cv::Mat &tvec)
{
    CV_Assert(fs.isOpened());
    CV_Assert(camera_matrix.type() == CV_64FC1 && camera_matrix.rows == 3 && camera_matrix.cols == 3);
    CV_Assert(dist_coeffs.type() == CV_64FC1 && dist_coeffs.rows == 1 && dist_coeffs.cols == 5);
    CV_Assert(rvec.type() == CV_64FC1 && rvec.rows == 3 && rvec.cols == 1);
    CV_Assert(tvec.type() == CV_64FC1 && tvec.rows == 3 && tvec.cols == 1);
   
    fs << "image-width" << camera_size.width;
    fs << "image-height" << camera_size.height;
    fs << "error" << error;
    fs << "camera-matrix" << camera_matrix;
    fs << "distortion-coefficients" << dist_coeffs;
    fs << "rvec" << rvec;
    fs << "tvec" << tvec;

    
    CV_Assert(fs.isOpened());
    return;
}

void fsiv_load_calibration_parameters(cv::FileStorage &fs,
                                      cv::Size &camera_size,
                                      float &error,
                                      cv::Mat &camera_matrix,
                                      cv::Mat &dist_coeffs,
                                      cv::Mat &rvec,
                                      cv::Mat &tvec)
{
    CV_Assert(fs.isOpened());
    
    fs["image-width"] >> camera_size.width;
    fs["image-height"] >> camera_size.height;
    fs["error"] >> error;
    fs["camera-matrix"] >> camera_matrix;
    fs["distortion-coefficients"] >> dist_coeffs;
    fs["rvec"] >> rvec;
    fs["tvec"] >> tvec;

    
    CV_Assert(fs.isOpened());
    CV_Assert(camera_matrix.type() == CV_64FC1 && camera_matrix.rows == 3 && camera_matrix.cols == 3);
    CV_Assert(dist_coeffs.type() == CV_64FC1 && dist_coeffs.rows == 1 && dist_coeffs.cols == 5);
    CV_Assert(rvec.type() == CV_64FC1 && rvec.rows == 3 && rvec.cols == 1);
    CV_Assert(tvec.type() == CV_64FC1 && tvec.rows == 3 && tvec.cols == 1);
    return;
}

void fsiv_undistort_image(const cv::Mat &input, cv::Mat &output,
                          const cv::Mat &camera_matrix,
                          const cv::Mat &dist_coeffs)
{
  
    cv::undistort(input, output, camera_matrix, dist_coeffs);
    
}

void fsiv_undistort_video_stream(cv::VideoCapture &input_stream,
                                 cv::VideoWriter &output_stream,
                                 const cv::Mat &camera_matrix,
                                 const cv::Mat &dist_coeffs,
                                 const int interp,
                                 const char *input_wname,
                                 const char *output_wname,
                                 double fps)
{
    CV_Assert(input_stream.isOpened());
    CV_Assert(output_stream.isOpened());
    
    cv::Mat frame, map1, map2;
    input_stream >> frame;

    cv::initUndistortRectifyMap(camera_matrix, dist_coeffs, cv::Mat(), 
                                    camera_matrix, frame.size(), CV_16SC2, map1, map2);
    
    while(input_stream.read(frame)){
        cv::Mat undistorted;

        cv::remap(frame, undistorted, map1, map2, interp);

        if(input_wname != nullptr){
            cv::imshow(input_wname, frame);
        }
        if(output_wname != nullptr){
            cv::imshow(output_wname, undistorted);
        }
        if(cv::waitKey(1) == 27)
            break;

        output_stream.write(undistorted);
    }
    //
    CV_Assert(input_stream.isOpened());
    CV_Assert(output_stream.isOpened());
}
