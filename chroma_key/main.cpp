#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "../chroma_key/common_code.hpp"


int main(int argc, char *argv[]){
    if(argc != 9){
        std::cerr << argv[0] << " input <foreground> - background <background> - hue <hue> - sensitivty <sensitivity>\n";
        return -1;
    }


    int hue = 0, sensitivity = 0;
    std::string foreground_file, background_file;

    for(int i = 1; i < argc; i++){
        if(std::string(argv[i]) == "-input"){
            foreground_file = argv[++i];
        }

        else if(std::string(argv[i]) == "-background"){
            background_file = argv[++i];
        }

        else if(std::string(argv[i]) == "-hue"){
            std::cout<<"hue input " << argv[i+1] << std::endl;

            try{
                hue = std::stoi(argv[++i]);

            }catch(std::invalid_argument& e){
                std::cerr<< "error\n";
                return -1;
            }
        }


        else if(std::string(argv[i]) =="-sensitivity"){
            std::cout<< "sensitivity input: "<< argv[i+1] << std::endl;
            try{
                sensitivity = std::stoi(argv[++i]);
            }catch(std::invalid_argument& e){
                std::cerr << "error";
                return -1;
            }
        }
    }


    std::cout << "foreground file "<< foreground_file<<"\n";
    std::cout << "background file "<< background_file<<"\n";
    std::cout << "hue  "<< hue<<"\n";
    std::cout << "sensitivty  "<< sensitivity <<"\n";


    cv::Mat foreground = cv::imread(foreground_file);
    cv::Mat background = cv::imread(background_file);


    if(foreground.empty() || background.empty()){
        std::cerr<<"error";
        return -1;
    }

    cv::Mat output = fsiv_apply_chroma_key(foreground, background, hue, sensitivity);

    if(output.empty()){
        std::cerr<<"error";
        return -1;
    }

    cv::imshow("output: ", output);
    cv::waitKey(0);


    return 0;


}

