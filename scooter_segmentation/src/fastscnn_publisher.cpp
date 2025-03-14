#include <cstdio>
#include <functional>
#include <iostream>
#include <tuple>

// #include "NNConverter.hpp"

#include "camera_info_manager/camera_info_manager.hpp"
#include "depthai_ros_msgs/msg/spatial_detection_array.hpp"
#include "rclcpp/node.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "stereo_msgs/msg/disparity_image.hpp"

// Inludes common necessary includes for development using depthai library
#include "depthai/depthai.hpp"
#include "depthai/device/DataQueue.hpp"
#include "depthai/device/Device.hpp"
#include "depthai/pipeline/Pipeline.hpp"
#include "depthai/pipeline/node/ColorCamera.hpp"
#include "depthai/pipeline/node/IMU.hpp"
#include "depthai/pipeline/node/MonoCamera.hpp"
#include "depthai/pipeline/node/SpatialDetectionNetwork.hpp"
#include "depthai/pipeline/node/StereoDepth.hpp"
#include "depthai/pipeline/node/XLinkIn.hpp"
#include "depthai/pipeline/node/XLinkOut.hpp"
#include "depthai_bridge/BridgePublisher.hpp"
#include "depthai_bridge/DisparityConverter.hpp"
#include "depthai_bridge/ImageConverter.hpp"
#include "depthai_bridge/NNConverter.hpp"
#include "depthai_bridge/ImuConverter.hpp"
#include "depthai_bridge/SpatialDetectionConverter.hpp"
#include "depthai_bridge/depthaiUtility.hpp"

std::vector<std::string> usbStrings = {"UNKNOWN", "LOW", "FULL", "HIGH", "SUPER", "SUPER_PLUS"};

dai::Pipeline createPipeline(bool enableDepth,
                                                   bool enablenn,
                                                   bool lrcheck,
                                                   bool extended,
                                                   bool subpixel,
                                                   bool rectify,
                                                   bool depth_aligned,
                                                   int stereo_fps,
                                                   int confidence,
                                                   int LRchecktresh,
                                                   int detectionClassesCount,
                                                   std::string stereoResolution,
                                                   std::string rgbResolutionStr,
                                                   int rgbScaleNumerator,
                                                   int rgbScaleDinominator,
                                                   int previewWidth,
                                                   int previewHeight,
                                                   bool syncNN,
                                                   std::string nnPath) {
    dai::Pipeline pipeline;

    pipeline.setOpenVINOVersion(dai::OpenVINO::Version::VERSION_2022_1);

    int rgbWidth, rgbHeight;
    
    auto camRgb = pipeline.create<dai::node::ColorCamera>();
    auto xoutRgb = pipeline.create<dai::node::XLinkOut>();
    xoutRgb->setStreamName("rgb");
    camRgb->setBoardSocket(dai::CameraBoardSocket::CAM_A);
    dai::node::ColorCamera::Properties::SensorResolution rgbResolution;

    if(rgbResolutionStr == "1080p") {
        rgbResolution = dai::node::ColorCamera::Properties::SensorResolution::THE_1080_P;
        rgbWidth = 1280;
        rgbHeight = 720;
        // 1920,1080
    } else if(rgbResolutionStr == "4K") {
        rgbResolution = dai::node::ColorCamera::Properties::SensorResolution::THE_4_K;
        rgbWidth = 3840;
        rgbHeight = 2160;
    } else if(rgbResolutionStr == "12MP") {
        rgbResolution = dai::node::ColorCamera::Properties::SensorResolution::THE_12_MP;
        rgbWidth = 4056;
        rgbHeight = 3040;
    } else if(rgbResolutionStr == "13MP") {
        rgbResolution = dai::node::ColorCamera::Properties::SensorResolution::THE_13_MP;
        rgbWidth = 4208;
        rgbHeight = 3120;
    } else {
        DEPTHAI_ROS_ERROR_STREAM("DEPTHAI", "Invalid parameter. -> rgbResolution: " << rgbResolutionStr);
        throw std::runtime_error("Invalid color camera resolution.");
    }

    camRgb->setResolution(rgbResolution);

    rgbWidth = rgbWidth * rgbScaleNumerator / rgbScaleDinominator;
    rgbHeight = rgbHeight * rgbScaleNumerator / rgbScaleDinominator;
    camRgb->setIspScale(rgbScaleNumerator, rgbScaleDinominator);

    camRgb->isp.link(xoutRgb->input);

    // std::cout << (rgbWidth % 2 == 0 && rgbHeight % 3 == 0) << std::endl;
    // assert(("Needs Width to be multiple of 2 and height to be multiple of 3 since the Image is NV12 format here.", (rgbWidth % 2 == 0 && rgbHeight % 3 ==
    // 0)));
    if(rgbWidth % 16 != 0) {
        if(rgbResolution == dai::node::ColorCamera::Properties::SensorResolution::THE_12_MP) {
            DEPTHAI_ROS_ERROR_STREAM("DEPTHAI",
                                        "RGB Camera width should be multiple of 16. Please choose a different scaling factor."
                                            << std::endl
                                            << "Here are the scalng options that works for 12MP with depth aligned" << std::endl
                                            << "4056 x 3040 *  2/13 -->  624 x  468" << std::endl
                                            << "4056 x 3040 *  2/39 -->  208 x  156" << std::endl
                                            << "4056 x 3040 *  2/51 -->  160 x  120" << std::endl
                                            << "4056 x 3040 *  4/13 --> 1248 x  936" << std::endl
                                            << "4056 x 3040 *  4/26 -->  624 x  468" << std::endl
                                            << "4056 x 3040 *  4/29 -->  560 x  420" << std::endl
                                            << "4056 x 3040 *  4/35 -->  464 x  348" << std::endl
                                            << "4056 x 3040 *  4/39 -->  416 x  312" << std::endl
                                            << "4056 x 3040 *  6/13 --> 1872 x 1404" << std::endl
                                            << "4056 x 3040 *  6/39 -->  624 x  468" << std::endl
                                            << "4056 x 3040 *  7/25 --> 1136 x  852" << std::endl
                                            << "4056 x 3040 *  8/26 --> 1248 x  936" << std::endl
                                            << "4056 x 3040 *  8/39 -->  832 x  624" << std::endl
                                            << "4056 x 3040 *  8/52 -->  624 x  468" << std::endl
                                            << "4056 x 3040 *  8/58 -->  560 x  420" << std::endl
                                            << "4056 x 3040 * 10/39 --> 1040 x  780" << std::endl
                                            << "4056 x 3040 * 10/59 -->  688 x  516" << std::endl
                                            << "4056 x 3040 * 12/17 --> 2864 x 2146" << std::endl
                                            << "4056 x 3040 * 12/26 --> 1872 x 1404" << std::endl
                                            << "4056 x 3040 * 12/39 --> 1248 x  936" << std::endl
                                            << "4056 x 3040 * 13/16 --> 3296 x 2470" << std::endl
                                            << "4056 x 3040 * 14/39 --> 1456 x 1092" << std::endl
                                            << "4056 x 3040 * 14/50 --> 1136 x  852" << std::endl
                                            << "4056 x 3040 * 14/53 --> 1072 x  804" << std::endl
                                            << "4056 x 3040 * 16/39 --> 1664 x 1248" << std::endl
                                            << "4056 x 3040 * 16/52 --> 1248 x  936" << std::endl);

        } else {
            DEPTHAI_ROS_ERROR_STREAM("DEPTHAI", "RGB Camera width should be multiple of 16. Please choose a different scaling factor.");
        }
        throw std::runtime_error("Adjust RGB Camaera scaling.");
    }

    

    if(enablenn) {
        if(previewWidth > rgbWidth or previewHeight > rgbHeight) {
            DEPTHAI_ROS_ERROR_STREAM("DEPTHAI",
                                        "Preview Image size should be smaller than the scaled resolution. Please adjust the "
                                        "scale parameters or the preview size accordingly.");
            throw std::runtime_error("Invalid Image Size");
        }

        camRgb->setColorOrder(dai::ColorCameraProperties::ColorOrder::BGR);
        camRgb->setInterleaved(false);
        camRgb->setPreviewSize(previewWidth, previewHeight);

        auto NN = pipeline.create<dai::node::NeuralNetwork>();
        NN->setBlobPath(nnPath);
        NN->setNumInferenceThreads(2);
        NN->input.setBlocking(false);

        auto xoutNN = pipeline.create<dai::node::XLinkOut>();
        auto xoutPreview = pipeline.create<dai::node::XLinkOut>();
        xoutPreview->setStreamName("preview");
        xoutNN->setStreamName("segmentation");

        // NN->setBlobPath(nnPath);
        // std::cout << nnPath << std::endl;
        camRgb->preview.link(NN->input);
        // xoutPreview->out.link(NN->input);
        if(syncNN)
            NN->passthrough.link(xoutPreview->input);
        else
            camRgb->preview.link(xoutPreview->input);
        NN->out.link(xoutNN->input);
        // stereo->depth.link(spatialDetectionNetwork->inputDepth);
    }

    return pipeline;
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("fastscnn_node");

    std::string tfPrefix, mode, mxId, resourceBaseFolder, nnPath;
    std::string monoResolution = "720p", rgbResolution = "1080p";
    int badParams = 0, stereo_fps, confidence, LRchecktresh, imuModeParam, detectionClassesCount, expTime, sensIso;
    int rgbScaleNumerator, rgbScaleDinominator, previewWidth, previewHeight;
    bool lrcheck, extended, subpixel, enableDepth, rectify, depth_aligned, manualExposure;
    bool enablenn, enableDotProjector, enableFloodLight;
    bool usb2Mode, poeMode, syncNN;
    double angularVelCovariance, linearAccelCovariance;
    double dotProjectorIntensity, floodLightIntensity;
    bool enableRosBaseTimeUpdate;
    std::string nnName(BLOB_NAME);  // Set your blob name for the model here
    bool useRGBSegmentation;
    std::string segmentationConfigPath;

    node->declare_parameter("mxId", "x");
    node->declare_parameter("usb2Mode", false);
    node->declare_parameter("poeMode", false);
    node->declare_parameter("resourceBaseFolder", resourceBaseFolder);

    node->declare_parameter("tf_prefix", "oak");
    node->declare_parameter("mode", "depth");
    node->declare_parameter("imuMode", 1);

    node->declare_parameter("lrcheck", true);
    node->declare_parameter("extended", false);
    node->declare_parameter("subpixel", true);
    node->declare_parameter("rectify", false);

    node->declare_parameter("depth_aligned", true);
    node->declare_parameter("stereo_fps", 30);
    node->declare_parameter("confidence", 200);
    node->declare_parameter("LRchecktresh", 5);
    node->declare_parameter("monoResolution", "720p");
    node->declare_parameter("rgbResolution", "1080p");
    node->declare_parameter("manualExposure", false);
    node->declare_parameter("expTime", 20000);
    node->declare_parameter("sensIso", 800);

    node->declare_parameter("rgbScaleNumerator", 2);
    node->declare_parameter("rgbScaleDinominator", 3);
    node->declare_parameter("previewWidth", 640);
    node->declare_parameter("previewHeight", 360);

    node->declare_parameter("angularVelCovariance", 0.02);
    node->declare_parameter("linearAccelCovariance", 0.0);
    node->declare_parameter("enablenn", true);
    node->declare_parameter("detectionClassesCount", 80);
    node->declare_parameter("syncNN", true);
    node->declare_parameter("nnName", "x");

    node->declare_parameter("enableDotProjector", false);
    node->declare_parameter("enableFloodLight", false);
    node->declare_parameter("dotProjectorIntensity", 0.5);
    node->declare_parameter("floodLightIntensity", 0.5);
    node->declare_parameter("enableRosBaseTimeUpdate", false);

    node->declare_parameter("useRGBSegmentation", false);
    node->declare_parameter("segmentationConfigPath", "");

    // updating parameters if defined in launch file.

    node->get_parameter("mxId", mxId);
    node->get_parameter("usb2Mode", usb2Mode);
    node->get_parameter("poeMode", poeMode);
    node->get_parameter("resourceBaseFolder", resourceBaseFolder);

    node->get_parameter("tf_prefix", tfPrefix);
    node->get_parameter("mode", mode);
    node->get_parameter("imuMode", imuModeParam);

    node->get_parameter("lrcheck", lrcheck);
    node->get_parameter("extended", extended);
    node->get_parameter("subpixel", subpixel);
    node->get_parameter("rectify", rectify);

    node->get_parameter("depth_aligned", depth_aligned);
    node->get_parameter("stereo_fps", stereo_fps);
    node->get_parameter("confidence", confidence);
    node->get_parameter("LRchecktresh", LRchecktresh);
    node->get_parameter("monoResolution", monoResolution);
    node->get_parameter("rgbResolution", rgbResolution);
    node->get_parameter("manualExposure", manualExposure);
    node->get_parameter("expTime", expTime);
    node->get_parameter("sensIso", sensIso);

    node->get_parameter("rgbScaleNumerator", rgbScaleNumerator);
    node->get_parameter("rgbScaleDinominator", rgbScaleDinominator);
    node->get_parameter("previewWidth", previewWidth);
    node->get_parameter("previewHeight", previewHeight);

    node->get_parameter("angularVelCovariance", angularVelCovariance);
    node->get_parameter("linearAccelCovariance", linearAccelCovariance);
    node->get_parameter("enablenn", enablenn);
    node->get_parameter("detectionClassesCount", detectionClassesCount);
    node->get_parameter("syncNN", syncNN);

    node->get_parameter("enableDotProjector", enableDotProjector);
    node->get_parameter("enableFloodLight", enableFloodLight);
    node->get_parameter("dotProjectorIntensity", dotProjectorIntensity);
    node->get_parameter("floodLightIntensity", floodLightIntensity);
    node->get_parameter("enableRosBaseTimeUpdate", enableRosBaseTimeUpdate);

    node->get_parameter("useRGBSegmentation", useRGBSegmentation);
    node->get_parameter("segmentationConfigPath", segmentationConfigPath);

    if(resourceBaseFolder.empty()) {
        throw std::runtime_error("Send the path to the resouce folder containing NNBlob in \'resourceBaseFolder\' ");
    }

    std::string nnParam;
    node->get_parameter("nnName", nnParam);
    if(nnParam != "x") {
        node->get_parameter("nnName", nnName);
    }
    nnPath = resourceBaseFolder + "/" + nnName;
    std::cout << nnPath << std::endl;
    if(mode == "depth") {
        enableDepth = true;
    } else {
        enableDepth = false;
    }

    dai::ros::ImuSyncMethod imuMode = static_cast<dai::ros::ImuSyncMethod>(imuModeParam);

    dai::Pipeline pipeline;
    int width, height;
    bool isDeviceFound = false;
    pipeline = createPipeline(enableDepth,
                            enablenn,
                            lrcheck,
                            extended,
                            subpixel,
                            rectify,
                            depth_aligned,
                            stereo_fps,
                            confidence,
                            LRchecktresh,
                            detectionClassesCount,
                            monoResolution,
                            rgbResolution,
                            rgbScaleNumerator,
                            rgbScaleDinominator,
                            previewWidth,
                            previewHeight,
                            syncNN,
                            nnPath);

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

    auto boardName = calibrationHandler.getEepromData().boardName;
    // if(height > 480 && boardName == "OAK-D-LITE" && depth_aligned == false) {
    //     width = 640;
    //     height = 480;
    // }
    std::vector<std::tuple<std::string, int, int>> irDrivers = device->getIrDrivers();
    if(!irDrivers.empty()) {
        if(enableDotProjector) {
            device->setIrLaserDotProjectorIntensity(dotProjectorIntensity);
        }

        if(enableFloodLight) {
            device->setIrFloodLightIntensity(floodLightIntensity);
        }
    }

  

    dai::rosBridge::ImageConverter rgbConverter(tfPrefix + "_rgb_camera_optical_frame", false);
    if(enableRosBaseTimeUpdate) {
        rgbConverter.setUpdateRosBaseTimeOnToRosMsg();
    }
   
    auto rgbCameraInfo = rgbConverter.calibrationToCameraInfo(calibrationHandler, dai::CameraBoardSocket::CAM_A, width, height);
    auto imgQueue = device->getOutputQueue("rgb", 30, false);
    // dai::rosBridge::ImageConverter rgbConverter(tfPrefix + "_rgb_camera_optical_frame", false);
    dai::rosBridge::BridgePublisher<sensor_msgs::msg::Image, dai::ImgFrame> rgbPublish(
        imgQueue,
        node,
        std::string("/oak_d_pro/rgb/image_raw"),
        std::bind(&dai::rosBridge::ImageConverter::toRosMsg, &rgbConverter, std::placeholders::_1, std::placeholders::_2),
        30,
        rgbCameraInfo,
        "/oak_d_pro/rgb");
    rgbPublish.addPublisherCallback();
    if(enablenn) {
        // auto previewQueue = device->getOutputQueue("preview", 30, false);
        dai::rosBridge::NNConverter nnConverter(
            tfPrefix + "_rgb_camera_optical_frame", 
            false, 
            false,
            useRGBSegmentation ? segmentationConfigPath : "");  // Pass config path only if RGB segmentation is enabled
        if(enableRosBaseTimeUpdate) {
            nnConverter.setUpdateRosBaseTimeOnToRosMsg();
        }
        auto previewQueue = device->getOutputQueue("preview", 30, false);
        auto segmentationQueue = device->getOutputQueue("segmentation", 30, false);
        // auto previewCameraInfo = rgbConverter.calibrationToCameraInfo(calibrationHandler, dai::CameraBoardSocket::CAM_A, previewWidth, previewHeight);

        // dai::rosBridge::BridgePublisher<sensor_msgs::msg::Image, dai::ImgFrame> previewPublish(
        //     previewQueue,
        //     node,
        //     std::string("/oak_d_pro/preview/image_raw"),
        //     std::bind(&dai::rosBridge::ImageConverter::toRosMsg, &rgbConverter, std::placeholders::_1, std::placeholders::_2),
        //     30,
        //     previewCameraInfo,
        //     "oak_d_pro/preview");
        // previewPublish.addPublisherCallback();

        dai::rosBridge::BridgePublisher<sensor_msgs::msg::Image, dai::NNData> segmentationPublish(
            segmentationQueue,
            node,
            std::string("/oak_d_pro/seg"),
            std::bind(&dai::rosBridge::NNConverter::toRosMsg,
                    &nnConverter,  // since the converter has the same frame name
                                        // and image type is also same we can reuse it
                    std::placeholders::_1,
                    std::placeholders::_2),
            30);

        segmentationPublish.addPublisherCallback();
        rclcpp::spin(node);
    }
    rclcpp::spin(node);
     

    return 0;
}


