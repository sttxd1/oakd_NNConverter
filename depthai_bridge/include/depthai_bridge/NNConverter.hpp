#pragma once

#include <deque>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <array>
#include <nlohmann/json.hpp>

#include "depthai/depthai.hpp"
#include "cv_bridge/cv_bridge.h"
#include "depthai-shared/common/CameraBoardSocket.hpp"
#include "depthai-shared/common/Point2f.hpp"
#include "depthai/device/CalibrationHandler.hpp"
#include "depthai/pipeline/datatype/EncodedFrame.hpp"
#include "depthai/pipeline/datatype/ImgFrame.hpp"
#include "ffmpeg_image_transport_msgs/msg/ffmpeg_packet.hpp"
#include "rclcpp/time.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/header.hpp"

namespace dai {

namespace ros {

namespace StdMsgs = std_msgs::msg;
namespace ImageMsgs = sensor_msgs::msg;

class NNConverter {
   public:
    NNConverter(const std::string frameName, bool interleaved, bool getBaseDeviceTimestamp = false, const std::string& jsonPath = "");
    ~NNConverter();
    /**
     * @brief Handles cases in which the ROS time shifts forward or backward
     *  Should be called at regular intervals or on-change of ROS time, depending
     *  on monitoring.
     *
     */
    void updateRosBaseTime();

    /**
     * @brief Commands the converter to automatically update the ROS base time on message conversion based on variable
     *
     * @param update: bool whether to automatically update the ROS base time on message conversion
     */
    void setUpdateRosBaseTimeOnToRosMsg(bool update = true) {
        updateRosBaseTimeOnToRosMsg = update;
    }

    /**
     * @brief Sets converter behavior to convert from bitstream to raw data.
     * @param srcType: The type of the bitstream data used for conversion.
     */
    // void convertFromBitstream(dai::RawImgFrame::Type srcType);

    /**
     * @brief Sets exposure offset when getting timestamps from the message.
     * @param offset: The exposure offset to be added to the timestamp.
     */
    void addExposureOffset(dai::CameraExposureOffset& offset);

    void toRosMsg(std::shared_ptr<dai::NNData> inData, std::deque<ImageMsgs::Image>& outImageMsgs);
    ImageMsgs::Image toRosMsgRawPtr(std::shared_ptr<dai::NNData> inData, const sensor_msgs::msg::CameraInfo& info = sensor_msgs::msg::CameraInfo(), int batch=1, int channels=1, int height=360, int width=640);

   private:
    std::vector<uint8_t> rearrangeData(const std::vector<int32_t>& inputData, int height, int width, int channels);
    std::vector<std::array<uint8_t, 3>> loadColorMap(const std::string& jsonPath);
    std::vector<uint8_t> rearrangeDataRGB(const std::vector<int32_t>& inputData, 
                                         int height, int width,
                                         const std::vector<std::array<uint8_t, 3>>& colorMap);
    // dai::RawImgFrame::Type _srcType;
    bool daiInterleaved;
    // bool c
    const std::string frameName = "";
    std::chrono::time_point<std::chrono::steady_clock> steadyBaseTime;

    rclcpp::Time rosBaseTime;
    bool getBaseDeviceTimestamp;
    // For handling ROS time shifts and debugging
    int64_t totalNsChange{0};
    // Whether to update the ROS base time on each message conversion
    bool updateRosBaseTimeOnToRosMsg{false};
    // dai::RawImgFrame::Type srcType;

    bool addExpOffset = false;
    dai::CameraExposureOffset expOffset;

    std::vector<std::array<uint8_t, 3>> colorMap;

};

}  // namespace ros

}  // namespace dai
