
#include <cstdio>
#include <iostream>

#include "camera_info_manager/camera_info_manager.hpp"
#include "depthai_bridge/BridgePublisher.hpp"
#include "depthai_bridge/ImageConverter.hpp"
#include "depthai_bridge/ImgDetectionConverter.hpp"
#include "rclcpp/executors.hpp"
#include "rclcpp/node.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"

// Inludes common necessary includes for development using depthai library
#include "depthai/device/DataQueue.hpp"
#include "depthai/device/Device.hpp"
#include "depthai/pipeline/Pipeline.hpp"
#include "depthai/pipeline/node/ColorCamera.hpp"
#include "depthai/pipeline/node/DetectionNetwork.hpp"
#include "depthai/pipeline/node/XLinkOut.hpp"

const std::vector<std::string> label_map = {
    "person",        "bicycle",      "car",           "motorbike",     "aeroplane",   "bus",         "train",       "truck",        "boat",
    "traffic light", "fire hydrant", "stop sign",     "parking meter", "bench",       "bird",        "cat",         "dog",          "horse",
    "sheep",         "cow",          "elephant",      "bear",          "zebra",       "giraffe",     "backpack",    "umbrella",     "handbag",
    "tie",           "suitcase",     "frisbee",       "skis",          "snowboard",   "sports ball", "kite",        "baseball bat", "baseball glove",
    "skateboard",    "surfboard",    "tennis racket", "bottle",        "wine glass",  "cup",         "fork",        "knife",        "spoon",
    "bowl",          "banana",       "apple",         "sandwich",      "orange",      "broccoli",    "carrot",      "hot dog",      "pizza",
    "donut",         "cake",         "chair",         "sofa",          "pottedplant", "bed",         "diningtable", "toilet",       "tvmonitor",
    "laptop",        "mouse",        "remote",        "keyboard",      "cell phone",  "microwave",   "oven",        "toaster",      "sink",
    "refrigerator",  "book",         "clock",         "vase",          "scissors",    "teddy bear",  "hair drier",  "toothbrush"};

dai::Pipeline createPipeline(bool syncNN, std::string nnPath) {
    dai::Pipeline pipeline;
    auto colorCam = pipeline.create<dai::node::ColorCamera>();
    auto detectionNetwork = pipeline.create<dai::node::YoloDetectionNetwork>();

    // create xlink connections
    auto xoutRgb = pipeline.create<dai::node::XLinkOut>();
    auto xoutNN = pipeline.create<dai::node::XLinkOut>();

    xoutRgb->setStreamName("preview");
    xoutNN->setStreamName("detections");

    // Properties
    colorCam->setPreviewSize(416, 416);
    colorCam->setResolution(dai::ColorCameraProperties::SensorResolution::THE_1080_P);
    colorCam->setInterleaved(false);
    colorCam->setColorOrder(dai::ColorCameraProperties::ColorOrder::BGR);
    colorCam->setFps(40);

    // Network specific settings
    detectionNetwork->setConfidenceThreshold(0.5f);
    detectionNetwork->setNumClasses(80);
    detectionNetwork->setCoordinateSize(4);
    detectionNetwork->setAnchors({10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319});
    detectionNetwork->setAnchorMasks({{"side26", {1, 2, 3}}, {"side13", {3, 4, 5}}});
    detectionNetwork->setIouThreshold(0.5f);
    detectionNetwork->setBlobPath(nnPath);
    detectionNetwork->setNumInferenceThreads(2);
    detectionNetwork->input.setBlocking(false);

    // Linking
    colorCam->preview.link(detectionNetwork->input);
    if(syncNN)
        detectionNetwork->passthrough.link(xoutRgb->input);
    else
        colorCam->preview.link(xoutRgb->input);

    detectionNetwork->out.link(xoutNN->input);
    return pipeline;
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("yolov4_node");

    std::string tfPrefix, resourceBaseFolder, nnPath;
    std::string camera_param_uri;
    std::string nnName(BLOB_NAME);  // Set your blob name for the model here
    bool syncNN;
    std::string monoResolution = "400p";

    node->declare_parameter("tf_prefix", "oak");
    node->declare_parameter("camera_param_uri", camera_param_uri);
    node->declare_parameter("sync_nn", true);
    node->declare_parameter("nnName", "");
    node->declare_parameter("resourceBaseFolder", "");

    node->get_parameter("tf_prefix", tfPrefix);
    node->get_parameter("camera_param_uri", camera_param_uri);
    node->get_parameter("sync_nn", syncNN);
    node->get_parameter("resourceBaseFolder", resourceBaseFolder);

    if(resourceBaseFolder.empty()) {
        throw std::runtime_error("Send the path to the resouce folder containing NNBlob in \'resourceBaseFolder\' ");
    }

    std::string nnParam;
    node->get_parameter("nnName", nnParam);
    if(nnParam != "x") {
        node->get_parameter("nnName", nnName);
    }

    nnPath = resourceBaseFolder + "/" + nnName;
    dai::Pipeline pipeline = createPipeline(syncNN, nnPath);
    dai::Device device(pipeline);

    auto colorQueue = device.getOutputQueue("preview", 30, false);
    auto detectionQueue = device.getOutputQueue("detections", 30, false);
    auto calibrationHandler = device.readCalibration();

    dai::rosBridge::ImageConverter rgbConverter(tfPrefix + "_rgb_camera_optical_frame", false);
    auto rgbCameraInfo = rgbConverter.calibrationToCameraInfo(calibrationHandler, dai::CameraBoardSocket::CAM_A, -1, -1);
    dai::rosBridge::BridgePublisher<sensor_msgs::msg::Image, dai::ImgFrame> rgbPublish(colorQueue,
                                                                                       node,
                                                                                       std::string("color/image"),
                                                                                       std::bind(&dai::rosBridge::ImageConverter::toRosMsg,
                                                                                                 &rgbConverter,  // since the converter has the same frame name
                                                                                                                 // and image type is also same we can reuse it
                                                                                                 std::placeholders::_1,
                                                                                                 std::placeholders::_2),
                                                                                       30,
                                                                                       rgbCameraInfo,
                                                                                       "color");

    dai::rosBridge::ImgDetectionConverter detConverter(tfPrefix + "_rgb_camera_optical_frame", 416, 416, false);
    dai::rosBridge::BridgePublisher<vision_msgs::msg::Detection2DArray, dai::ImgDetections> detectionPublish(
        detectionQueue,
        node,
        std::string("color/yolov4_detections"),
        std::bind(&dai::rosBridge::ImgDetectionConverter::toRosMsg, &detConverter, std::placeholders::_1, std::placeholders::_2),
        30);

    detectionPublish.addPublisherCallback();
    rgbPublish.addPublisherCallback();  // addPublisherCallback works only when the dataqueue is non blocking.

    rclcpp::spin(node);

    return 0;
}




































#include <cstdio>
#include <functional>
#include <iostream>
#include <tuple>

#include "camera_info_manager/camera_info_manager.hpp"
#include "depthai_ros_msgs/msg/spatial_detection_array.hpp"
#include "rclcpp/node.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/imu.hpp"

#include "depthai/depthai.hpp"
#include "depthai/device/DataQueue.hpp"
#include "depthai/device/Device.hpp"
#include "depthai/pipeline/Pipeline.hpp"
#include "depthai/pipeline/node/ColorCamera.hpp"
#include "depthai/pipeline/node/IMU.hpp"
#include "depthai/pipeline/node/XLinkOut.hpp"
#include "depthai_bridge/BridgePublisher.hpp"
#include "depthai_bridge/ImageConverter.hpp"
#include "depthai_bridge/NNConverter.hpp"
#include "depthai_bridge/ImuConverter.hpp"
#include "depthai_bridge/depthaiUtility.hpp"

std::vector<std::string> usbStrings = {"UNKNOWN", "LOW", "FULL", "HIGH", "SUPER", "SUPER_PLUS"};

// std::tuple<dai::Pipeline, int, int> createPipeline(bool enablenn,
//                                                    std::string rgbResolutionStr,
//                                                    int rgbScaleNumerator,
//                                                    int rgbScaleDinominator,
//                                                    int previewWidth,
//                                                    int previewHeight,
//                                                    bool syncNN,
//                                                    std::string nnPath) {
//     dai::Pipeline pipeline;

//     pipeline.setOpenVINOVersion(dai::OpenVINO::Version::VERSION_2022_1);

//     // RGB Camera
//     auto camRgb = pipeline.create<dai::node::ColorCamera>();
//     auto xoutRgb = pipeline.create<dai::node::XLinkOut>();
//     xoutRgb->setStreamName("rgb");
//     camRgb->setBoardSocket(dai::CameraBoardSocket::CAM_A);

//     dai::node::ColorCamera::Properties::SensorResolution rgbResolution;
//     int rgbWidth, rgbHeight;

//     if(rgbResolutionStr == "720p") {
//         rgbResolution = dai::node::ColorCamera::Properties::SensorResolution::THE_720_P;
//         rgbWidth = 1280;
//         rgbHeight = 720;
//     } else if(rgbResolutionStr == "4K") {
//         rgbResolution = dai::node::ColorCamera::Properties::SensorResolution::THE_4_K;
//         rgbWidth = 3840;
//         rgbHeight = 2160;
//     } else if(rgbResolutionStr == "12MP") {
//         rgbResolution = dai::node::ColorCamera::Properties::SensorResolution::THE_12_MP;
//         rgbWidth = 4056;
//         rgbHeight = 3040;
//     } else if(rgbResolutionStr == "13MP") {
//         rgbResolution = dai::node::ColorCamera::Properties::SensorResolution::THE_13_MP;
//         rgbWidth = 4208;
//         rgbHeight = 3120;
//     } else {
//         DEPTHAI_ROS_ERROR_STREAM("DEPTHAI", "Invalid parameter. -> rgbResolution: " << rgbResolutionStr);
//         throw std::runtime_error("Invalid color camera resolution.");
//     }

//     camRgb->setResolution(rgbResolution);
//     // rgbWidth = rgbWidth * rgbScaleNumerator / rgbScaleDinominator;
//     // rgbHeight = rgbHeight * rgbScaleNumerator / rgbScaleDinominator;
//     // camRgb->setIspScale(rgbScaleNumerator, rgbScaleDinominator);
//     // camRgb->isp.link(xoutRgb->input);

//     // if(rgbWidth % 16 != 0) {
//     //     DEPTHAI_ROS_ERROR_STREAM("DEPTHAI", "RGB Camera width should be multiple of 16. Please choose a different scaling factor.");
//     //     throw std::runtime_error("Adjust RGB Camera scaling.");
//     // }

//     if(enablenn) {
//         if(previewWidth > rgbWidth || previewHeight > rgbHeight) {
//             DEPTHAI_ROS_ERROR_STREAM("DEPTHAI",
//                                      "Preview Image size should be smaller than the scaled resolution. Please adjust the "
//                                      "scale parameters or the preview size accordingly.");
//             throw std::runtime_error("Invalid Image Size");
//         }

//         camRgb->setColorOrder(dai::ColorCameraProperties::ColorOrder::BGR);
//         camRgb->setInterleaved(false);
//         camRgb->setPreviewSize(previewWidth, previewHeight);

//         auto NN = pipeline.create<dai::node::NeuralNetwork>();
//         NN->setBlobPath(nnPath);
//         NN->setNumInferenceThreads(2);
//         NN->input.setBlocking(false);

//         auto xoutNN = pipeline.create<dai::node::XLinkOut>();
//         auto xoutPreview = pipeline.create<dai::node::XLinkOut>();
//         xoutPreview->setStreamName("preview");
//         xoutNN->setStreamName("segmentation");

//         camRgb->preview.link(NN->input);
//         if(syncNN)
//             NN->passthrough.link(xoutPreview->input);
//         else
//             camRgb->preview.link(xoutPreview->input);
//         NN->out.link(xoutNN->input);
//     }

//     return std::make_tuple(pipeline, rgbWidth, rgbHeight);
// }

dai::Pipeline createPipeline(bool syncNN, std::string nnPath) {
    dai::Pipeline pipeline;
    auto colorCam = pipeline.create<dai::node::ColorCamera>();
    // auto detectionNetwork = pipeline.create<dai::node::YoloDetectionNetwork>();

    // create xlink connections
    auto xoutRgb = pipeline.create<dai::node::XLinkOut>();
    // auto xoutNN = pipeline.create<dai::node::XLinkOut>();

    xoutRgb->setStreamName("rgb");
    // xoutNN->setStreamName("detections");

    // Properties
    colorCam->setPreviewSize(416, 416);
    colorCam->setResolution(dai::ColorCameraProperties::SensorResolution::THE_1080_P);
    colorCam->setInterleaved(false);
    colorCam->setColorOrder(dai::ColorCameraProperties::ColorOrder::BGR);
    colorCam->setFps(40);

    // Network specific settings
    // detectionNetwork->setConfidenceThreshold(0.5f);
    // detectionNetwork->setNumClasses(80);
    // detectionNetwork->setCoordinateSize(4);
    // detectionNetwork->setAnchors({10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319});
    // detectionNetwork->setAnchorMasks({{"side26", {1, 2, 3}}, {"side13", {3, 4, 5}}});
    // detectionNetwork->setIouThreshold(0.5f);
    // detectionNetwork->setBlobPath(nnPath);
    // detectionNetwork->setNumInferenceThreads(2);
    // detectionNetwork->input.setBlocking(false);

    // // Linking
    // colorCam->preview.link(detectionNetwork->input);
    // if(syncNN)
    //     detectionNetwork->passthrough.link(xoutRgb->input);
    // else
    //     colorCam->preview.link(xoutRgb->input);

    // detectionNetwork->out.link(xoutNN->input);
    return pipeline;
}


int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("fastscnn_node");

    std::string tfPrefix, mxId, resourceBaseFolder, nnPath;
    std::string rgbResolution = "720p";
    int rgbScaleNumerator, rgbScaleDinominator, previewWidth, previewHeight;
    bool enablenn, usb2Mode, poeMode, syncNN;
    bool enableRosBaseTimeUpdate;
    std::string nnName(BLOB_NAME);  // Set your blob name for the model here
    bool useRGBSegmentation;
    std::string segmentationConfigPath;

    node->declare_parameter("mxId", "x");
    node->declare_parameter("usb2Mode", false);
    node->declare_parameter("poeMode", false);
    node->declare_parameter("resourceBaseFolder", resourceBaseFolder);
    node->declare_parameter("tf_prefix", "oak");
    node->declare_parameter("rgbResolution", "720p");
    node->declare_parameter("rgbScaleNumerator", 1);
    node->declare_parameter("rgbScaleDinominator", 1);
    node->declare_parameter("previewWidth", 640);
    node->declare_parameter("previewHeight", 360);
    node->declare_parameter("enablenn", true);
    node->declare_parameter("syncNN", true);
    node->declare_parameter("nnName", "x");
    node->declare_parameter("enableRosBaseTimeUpdate", false);
    node->declare_parameter("useRGBSegmentation", false);
    node->declare_parameter("segmentationConfigPath", "");

    node->get_parameter("mxId", mxId);
    node->get_parameter("usb2Mode", usb2Mode);
    node->get_parameter("poeMode", poeMode);
    node->get_parameter("resourceBaseFolder", resourceBaseFolder);
    node->get_parameter("tf_prefix", tfPrefix);
    node->get_parameter("rgbResolution", rgbResolution);
    node->get_parameter("rgbScaleNumerator", rgbScaleNumerator);
    node->get_parameter("rgbScaleDinominator", rgbScaleDinominator);
    node->get_parameter("previewWidth", previewWidth);
    node->get_parameter("previewHeight", previewHeight);
    node->get_parameter("enablenn", enablenn);
    node->get_parameter("syncNN", syncNN);
    node->get_parameter("enableRosBaseTimeUpdate", enableRosBaseTimeUpdate);
    node->get_parameter("useRGBSegmentation", useRGBSegmentation);
    node->get_parameter("segmentationConfigPath", segmentationConfigPath);

    if(resourceBaseFolder.empty()) {
        throw std::runtime_error("Send the path to the resource folder containing NNBlob in \'resourceBaseFolder\' ");
    }

    std::string nnParam;
    node->get_parameter("nnName", nnParam);
    if(nnParam != "x") {
        node->get_parameter("nnName", nnName);
    }
    nnPath = resourceBaseFolder + "/" + nnName;
    std::cout << nnPath << std::endl;

    // dai::Pipeline pipeline;
    // int width, height;
    bool isDeviceFound = false;
    // std::tie(pipeline, width, height) = createPipeline(enablenn,
    //                                                    rgbResolution,
    //                                                    rgbScaleNumerator,
    //                                                    rgbScaleDinominator,
    //                                                    previewWidth,
    //                                                    previewHeight,
    //                                                    syncNN,
    //                                                    nnPath);
    dai::Pipeline pipeline = createPipeline(syncNN, nnPath);

    std::shared_ptr<dai::Device> device;
    std::vector<dai::DeviceInfo> availableDevices = dai::Device::getAllAvailableDevices();

    std::cout << "Listing available devices..." << std::endl;
    for(auto deviceInfo : availableDevices) {
        std::cout << "Device Mx ID: " << deviceInfo.getMxId() << std::endl;
        if(deviceInfo.getMxId() == mxId) {
            if(deviceInfo.state == X_LINK_UNBOOTED || deviceInfo.state == X_LINK_BOOTLOADER) {
                isDeviceFound = true;
                if(poeMode) {
                    device = std::make_shared<dai::Device>(pipeline, deviceInfo);
                } else {
                    device = std::make_shared<dai::Device>(pipeline, deviceInfo, usb2Mode);
                }
                break;
            } else if(deviceInfo.state == X_LINK_BOOTED) {
                throw std::runtime_error("\" DepthAI Device with MxId  \"" + mxId + "\" is already booted on different process.  \"");
            }
        } else if(mxId == "x") {
            isDeviceFound = true;
            device = std::make_shared<dai::Device>(pipeline);
        }
    }
    if(!isDeviceFound) {
        throw std::runtime_error("\" DepthAI Device with MxId  \"" + mxId + "\" not found.  \"");
    }

    if(!poeMode) {
        std::cout << "Device USB status: " << usbStrings[static_cast<int32_t>(device->getUsbSpeed())] << std::endl;
    }

    auto calibrationHandler = device->readCalibration();
    std::cout << "Calibration read" << std::endl;

    dai::rosBridge::ImageConverter rgbConverter(tfPrefix + "_rgb_camera_optical_frame", false);
    if(enableRosBaseTimeUpdate) {
        rgbConverter.setUpdateRosBaseTimeOnToRosMsg();
    }
    std::cout << "RGB Converter created" << std::endl;

    auto rgbCameraInfo = rgbConverter.calibrationToCameraInfo(calibrationHandler, dai::CameraBoardSocket::CAM_A, -1, -1);
    std::cout << "Camera info created" << std::endl;

    // std::cout << "Before getOutputQueue rgb" << std::endl;
    auto imgQueue = device->getOutputQueue("rgb", 30, false);
    // std::cout << "RGB queue created" << std::endl;

    // auto frame = imgQueue->tryGet<dai::ImgFrame>();
    // if(frame) {
    //     std::cout << "RGB frame received: " << frame->getWidth() << "x" << frame->getHeight() << std::endl;
    // } else {
    //     std::cout << "No RGB frame available" << std::endl;
    // }

    dai::rosBridge::BridgePublisher<sensor_msgs::msg::Image, dai::ImgFrame> rgbPublish(
        imgQueue,
        node,
        std::string("oak_d_pro/rgb/image_raw"),
        std::bind(&dai::rosBridge::ImageConverter::toRosMsg, &rgbConverter, std::placeholders::_1, std::placeholders::_2),
        30,
        rgbCameraInfo,
        "oak_d_pro/rgb");
    rgbPublish.addPublisherCallback();

    // if(enablenn) {
    //     dai::rosBridge::NNConverter nnConverter(tfPrefix + "_rgb_camera_optical_frame", false, false, useRGBSegmentation ? segmentationConfigPath : "");
    //     if(enableRosBaseTimeUpdate) {
    //         nnConverter.setUpdateRosBaseTimeOnToRosMsg();
    //     }
    //     auto previewQueue = device->getOutputQueue("preview", 30, false);
    //     auto segmentationQueue = device->getOutputQueue("segmentation", 30, false);

    //     dai::rosBridge::BridgePublisher<sensor_msgs::msg::Image, dai::NNData> segmentationPublish(
    //         segmentationQueue,
    //         node,
    //         std::string("/oak_d_pro/seg"),
    //         std::bind(&dai::rosBridge::NNConverter::toRosMsg, &nnConverter, std::placeholders::_1, std::placeholders::_2),
    //         30);
    //     segmentationPublish.addPublisherCallback();

    //     // // Optional: Publish preview images
    //     // dai::rosBridge::BridgePublisher<sensor_msgs::msg::Image, dai::ImgFrame> previewPublish(
    //     //     previewQueue,
    //     //     node,
    //     //     std::string("/oak_d_pro/preview/image_raw"),
    //     //     std::bind(&dai::rosBridge::ImageConverter::toRosMsg, &rgbConverter, std::placeholders::_1, std::placeholders::_2),
    //     //     30,
    //     //     rgbCameraInfo,
    //     //     "/oak_d_pro/preview");
    //     // previewPublish.addPublisherCallback();
    // }

    rclcpp::spin(node);
    return 0;
}