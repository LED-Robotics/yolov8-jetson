//
// Created by ubuntu on 3/16/23.
//
#include "chrono"
#include "opencv2/opencv.hpp"
#include "yolov8.hpp"

const std::vector<std::string> CLASS_NAMES = {
    "note",         "robot"};

const std::vector<std::vector<unsigned int>> COLORS = {
    {255, 0, 0},   {0, 255, 0}};

int main(int argc, char** argv)
{
    const std::string engine_file_path{argv[1]};
    const std::string path{argv[2]};

    assert(argc == 3);

    auto yolov8 = new YOLOv8(engine_file_path);
    yolov8->make_pipe(true);

    cv::Mat             res, image;
    cv::Size            size = cv::Size{640, 640};
    std::vector<Object> objs;

    cv::namedWindow("result", cv::WINDOW_AUTOSIZE);

    cv::VideoCapture cap(path);

    if (!cap.isOpened()) {
        printf("can not open %s\n", path.c_str());
        return -1;
    }
    while (cap.read(image)) {
        objs.clear();
        yolov8->copy_from_Mat(image, size);
        auto start = std::chrono::system_clock::now();
        yolov8->infer();
        auto end = std::chrono::system_clock::now();
        yolov8->postprocess(objs);
        yolov8->draw_objects(image, res, objs, CLASS_NAMES, COLORS);
        auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
        printf("cost %2.4lf ms\n", tc);
        cv::imshow("result", res);
        if (cv::waitKey(10) == 'q') {
           break;
        }
    }
    cv::destroyAllWindows();
    delete yolov8;
    return 0;
}
