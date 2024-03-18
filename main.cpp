#include <iostream>
#include <filesystem>
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

std::string getNewFileName() {
    std::string path = "./videos";
    int num = 0;
    for (const auto & entry : std::filesystem::directory_iterator(path)) {
        auto name = entry.path().generic_string();
        std::cout << name << std::endl;
        auto aviPos = name.find(".avi");
        if(aviPos != std::string::npos) {
            auto startPos = name.find("_") + 1;
            auto numStr = name.substr(startPos, aviPos - startPos);
            std::cout << "numstr: " << numStr << '\n';
            int currentNum = std::stoi(numStr);
            if(currentNum >= num) num = currentNum + 1;
        }
    }
    return "./videos/output_" + std::to_string(num) + ".avi";
}

int main(int argc, char** argv)
{
    int frameTime = 0;
    double trackedSize = 0.0;
    int trackedIndex = 0;
    double tx = 0.0;
    double ty = 0.0;
    bool targetFound = false;
    int width, height;
    width = 640;
    height = 640;
    cs::VideoMode startingMode{cs::VideoMode::PixelFormat::kMJPEG, width, height, 30};
    // const std::string engine_file_path{argv[1]};
    // const std::string path{argv[2]};

    cs::UsbCamera intakeCam{"intakeCam", "/dev/video0"};
    intakeCam.SetVideoMode(startingMode);
    auto inst = nt::NetworkTableInstance::GetDefault();
    inst.SetServerTeam(6722);
    inst.StartClient4("jetson-client");
    auto table = inst.GetTable("/jetson");
    cs::CvSink intakeSink{frc::CameraServer::GetVideo(intakeCam)};
    assert(argc == 3);

    auto yolov8 = new YOLOv8("best.engine");
    yolov8->make_pipe(true);

    cv::Mat             res, image;
    cv::Size            size = cv::Size{width, height};
    std::vector<Object> objs;

    cs::CvSource intakeSource{"intakeRes", startingMode};
    frc::CameraServer::StartAutomaticCapture(intakeSource);

    int time = intakeSink.GrabFrameNoTimeout(image);
    cv::VideoWriter outputVideo{getNewFileName(), cv::VideoWriter::fourcc('M','J','P','G'), 30, {image.size().width, image.size().height}};
    if(!outputVideo.isOpened()) {
        std::cout << "Video not opened!\n";
    } else {
        std::cout << "Video opened!\n";
    }
    // for(int i = 0; i < 150; i++) {
    while(true) {
        frameTime = intakeSink.GrabFrameNoTimeout(image);
        outputVideo.write(image);
        objs.clear();
        yolov8->copy_from_Mat(image, size);
        yolov8->infer();
        yolov8->postprocess(objs);
        frameTime = 0;
        trackedSize = 0.0;
        trackedIndex = 0;
        tx = 0.0;
        ty = 0.0;
        targetFound = false;
        // target processing
        for(int i = 0; i < objs.size(); i++) {
            bool targetViable = true;
            if(objs[i].label != 0) continue;
            double adjX = objs[i].rect.x - width / 2 + objs[i].rect.width / 2;
            double adjY = height / 2 - objs[i].rect.y - objs[i].rect.height;
           //targetViable &= adjY < 50.0;
            double size = objs[i].rect.width * objs[i].rect.height;
            size = size / (width * height);
            if(!targetViable) continue;
            targetFound = true;
            if(size > trackedSize) {
                trackedSize = size;
                trackedIndex = i;
                tx = adjX;
                ty = adjY;
            }
        }
        // send data to network table
        table->PutNumber("tx", tx);
        table->PutNumber("ty", ty);
        table->PutNumber("ts", trackedSize);
        table->PutBoolean("tv", targetFound);
        if(targetFound) {
            printf("(%f, %f) Size: %f\n", tx, ty, trackedSize);
        } else {
            printf("No viable target!\n");
        }

        yolov8->draw_objects(image, res, objs, CLASS_NAMES, COLORS);
        intakeSource.PutFrame(res);

    }
    outputVideo.release();
    delete yolov8;
    return 0;
}
