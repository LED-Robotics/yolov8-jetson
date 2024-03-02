//
// Created by ubuntu on 3/16/23.
//
#include "chrono"
#include "opencv2/opencv.hpp"
#include "yolov8.hpp"
#include <networktables/NetworkTableInstance.h>
#include <networktables/NetworkTable.h>
#include "cameraserver/CameraServer.h"

const std::vector<std::string> CLASS_NAMES = {
    "note",         "robot"};

const std::vector<std::vector<unsigned int>> COLORS = {
    {255, 0, 0},   {0, 255, 0}};

int main(int argc, char** argv)
{
    int width, height;
    width = 640;
    height = 640;
    cs::VideoMode startingMode{cs::VideoMode::PixelFormat::kMJPEG, width, height, 30};
    const std::string engine_file_path{argv[1]};
    const std::string path{argv[2]};

    cs::UsbCamera intakeCam{"intakeCam", path};
    intakeCam.SetVideoMode(startingMode);
    // auto inst = nt::NetworkTableInstance::GetDefault();
    // auto table = inst.GetTable("orin");
    // inst.SetServerTeam(6722);
    cs::CvSink intakeSink{frc::CameraServer::GetVideo(intakeCam)};
    assert(argc == 3);

    auto yolov8 = new YOLOv8(engine_file_path);
    yolov8->make_pipe(true);

    cv::Mat             res, image;
    cv::Size            size = cv::Size{width, height};
    std::vector<Object> objs;

    cs::CvSource intakeSource{"intakeRes", startingMode};
    frc::CameraServer::StartAutomaticCapture(intakeSource);

    cv::namedWindow("result", cv::WINDOW_AUTOSIZE);
    int frameTime = 0;
    while(true) {
        frameTime = intakeSink.GrabFrameNoTimeout(image);
        objs.clear();
        yolov8->copy_from_Mat(image, size);
        auto start = std::chrono::system_clock::now();
        yolov8->infer();
        auto end = std::chrono::system_clock::now();
        yolov8->postprocess(objs);
        yolov8->draw_objects(image, res, objs, CLASS_NAMES, COLORS);
        auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
        printf("cost %2.4lf ms\n", tc);
        intakeSource.PutFrame(res);
    }
    cv::destroyAllWindows();
    delete yolov8;
    return 0;
}
