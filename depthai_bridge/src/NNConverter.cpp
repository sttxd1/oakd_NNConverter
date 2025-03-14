#include "depthai_bridge/NNConverter.hpp"

#include <sensor_msgs/msg/detail/compressed_image__struct.hpp>

#include "depthai/depthai.hpp"
#include "depthai-shared/datatype/RawEncodedFrame.hpp"
#include "depthai/pipeline/datatype/EncodedFrame.hpp"
#include "depthai_bridge/depthaiUtility.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include <fstream>
#include <nlohmann/json.hpp>

namespace dai {

namespace ros {

NNConverter::NNConverter(const std::string frameName, bool interleaved, bool getBaseDeviceTimestamp, const std::string& jsonPath)
    : frameName(frameName), 
      daiInterleaved(interleaved), 
      steadyBaseTime(std::chrono::steady_clock::now()), 
      getBaseDeviceTimestamp(getBaseDeviceTimestamp) {
    rosBaseTime = rclcpp::Clock().now();
    if (!jsonPath.empty()) {
        colorMap = loadColorMap(jsonPath);
        std::cout << "USE JSON" << std::endl;
    }
}

NNConverter::~NNConverter() = default;

void NNConverter::updateRosBaseTime() {
    updateBaseTime(steadyBaseTime, rosBaseTime, totalNsChange);
}

std::vector<uint8_t> NNConverter::rearrangeData(const std::vector<int32_t>& inputData, int height, int width, int channels) {
    if (inputData.size() != static_cast<size_t>(height * width * channels)) {
        throw std::runtime_error("Input data size does not match the provided dimensions.");
    }
    std::vector<uint8_t> outputData(height * width * channels);
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            for (int c = 0; c < channels; ++c) {
                // Row-major order: output[h][w][c] = input[c][h][w]
                // Convert int32_t to uint8_t during assignment (ensure valid range [0, 255])
                int32_t value = inputData[c * height * width + h * width + w];
                outputData[h * width * channels + w * channels + c] = static_cast<uint8_t>(std::clamp(value, 0, 255));
            }
        }
    }
    return outputData;
}

std::vector<uint8_t> NNConverter::rearrangeDataRGB(const std::vector<int32_t>& inputData, 
                                                  int height, int width,
                                                  const std::vector<std::array<uint8_t, 3>>& colorMap) {
    if (inputData.size() != static_cast<size_t>(height * width)) {
        throw std::runtime_error("Input data size does not match the provided dimensions.");
    }
    
    std::vector<uint8_t> outputData(height * width * 3);
    
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            int32_t classIdx = inputData[h * width + w];
            
            if (classIdx >= 0 && classIdx < static_cast<int>(colorMap.size())) {
                const auto& color = colorMap[classIdx];
                int outputIdx = (h * width + w) * 3;
                outputData[outputIdx] = color[0];     // R
                outputData[outputIdx + 1] = color[1]; // G
                outputData[outputIdx + 2] = color[2]; // B
            }
        }
    }
    
    return outputData;
}

void NNConverter::addExposureOffset(dai::CameraExposureOffset& offset) {
    expOffset = offset;
    addExpOffset = true;
}

ImageMsgs::Image NNConverter::toRosMsgRawPtr(std::shared_ptr<dai::NNData> inData, 
    const sensor_msgs::msg::CameraInfo& info, int batch, int channels, int height, int width) {
    
    if(updateRosBaseTimeOnToRosMsg) {
        updateRosBaseTime();
    }
    
    ImageMsgs::Image outImageMsg;
    outImageMsg.header.stamp = rclcpp::Clock().now();
    outImageMsg.header.frame_id = info.header.frame_id;
    
    std::string layerName = "output";
    std::vector<int> data = inData->getLayerInt32(layerName);
    // std::cout << "USE data" << std::endl;
    
    std::vector<uint8_t> rearrangedData;
    if (!colorMap.empty()) {
        rearrangedData = rearrangeDataRGB(data, height, width, colorMap);
        outImageMsg.encoding = "rgb8";
        outImageMsg.step = width * 3;
    } else {
        rearrangedData = rearrangeData(data, height, width, 1);
        outImageMsg.encoding = "mono8";
        outImageMsg.step = width;
    }
    
    outImageMsg.height = height;
    outImageMsg.width = width;
    outImageMsg.is_bigendian = false;
    outImageMsg.data = std::move(rearrangedData);
    
    return outImageMsg;
}

void NNConverter::toRosMsg(std::shared_ptr<dai::NNData> inData, std::deque<ImageMsgs::Image>& outImageMsgs) {
    sensor_msgs::msg::CameraInfo info;
    info.header.frame_id = frameName;
    // std::cout << "USE NIMA" << std::endl;
    auto outImageMsg = toRosMsgRawPtr(inData, info, 1, 1, 360, 640);
    outImageMsgs.push_back(outImageMsg);
}

std::vector<std::array<uint8_t, 3>> NNConverter::loadColorMap(const std::string& jsonPath) {
    std::vector<std::array<uint8_t, 3>> result;
    try {
        std::ifstream file(jsonPath);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open JSON file: " + jsonPath);
        }
        nlohmann::json config;
        file >> config;
        
        for (const auto& label : config["labels"]) {
            std::array<uint8_t, 3> color = {
                static_cast<uint8_t>(label["color"][0]),
                static_cast<uint8_t>(label["color"][1]),
                static_cast<uint8_t>(label["color"][2])
            };
            result.push_back(color);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error loading color map: " << e.what() << std::endl;
    }
    return result;
}
}  // namespace ros
}  // namespace dai

