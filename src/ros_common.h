#ifndef ROS_COMMON_H
#define ROS_COMMON_H

#include <cv_bridge/cv_bridge.h>
#include <pcl_conversions/pcl_conversions.h>
#ifdef ROS_VERSION_1

#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <image_transport/image_transport.h>
#include <nav_msgs/Path.h>

using RosNode = ros::NodeHandle;

using RosHeader = std_msgs::Header;
using RosImu = sensor_msgs::Imu;
using RosImuPtr = sensor_msgs::Imu::Ptr;
using RosImuConstPtr = sensor_msgs::Imu::ConstPtr;
using RosCloud = sensor_msgs::PointCloud2;
using RosCloudConstPtr = sensor_msgs::PointCloud2::ConstPtr;
using RosImage = sensor_msgs::Image;
using RosImagePtr = sensor_msgs::Image::Ptr;
using RosImageConstPtr = sensor_msgs::Image::ConstPtr;
using RosCompressedImage = sensor_msgs::CompressedImage;
using RosCompressedImageConstPtr = sensor_msgs::CompressedImage::ConstPtr;
using RosPath = nav_msgs::Path;
using RosPoseStamped = geometry_msgs::PoseStamped;
using RosTransformStamped = geometry_msgs::TransformStamped;
using RosTfQuaternion = tf::Quaternion;

using RosImuSubscriber = ros::Subscriber;
using RosCloudSubscriber = ros::Subscriber;
using RosImgSubscriber = ros::Subscriber;
using RosCompressedImgSubscriber = ros::Subscriber;

using RosCloudPublisher = ros::Publisher;
using RosPathPublisher = ros::Publisher;
using RosTransformBroadcaster = tf::TransformBroadcaster;

#define ROS_INIT(argc, argv, name) ros::init(argc, argv, name)
#define ROS_NODE(name) ros::NodeHandle()
#define ROS_SPIN(node) ros::spin()
#define ROS_SHUTDOWN() ros::shutdown()
#define ROS_OK() ros::ok()
#define ROS_TIME_NOW() ros::Time::now()
#define ROS_TIMESTAMP_TO_SEC(ros_time) ros_time.toSec()
void SetRosTimestamp(RosHeader &header, double sec);

#define ROS_ACCESS(node, method) node.method
#define ROS_SUBSCRIBE(node, type, topic, queue, callback) node.subscribe<type>(topic, queue, callback)
#define ROS_SUBSCRIBE_MEMBER(node, type, topic, queue, callback, this) node.subscribe<type>(topic, queue, callback, this)
#define ROS_ADVERTISE(node, type, topic, queue) node.advertise<type>(topic, queue)
#define ROS_GET_SUB_NUM(pub) pub.getNumSubscribers()
#define ROS_PUBLISH(pub, msg) pub.publish(msg)

#define ROS_TRANSFORM_BROADCASTER(node) tf::TransformBroadcaster()
#define ROS_SEND_TRANSFORM(pub, msg) pub.sendTransform(msg)


#elif defined(ROS_VERSION_2)

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/header.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <image_transport/image_transport.hpp>
#include "nav_msgs/msg/path.hpp"

using RosNode = rclcpp::Node::SharedPtr;

using RosHeader = std_msgs::msg::Header;
using RosImu = sensor_msgs::msg::Imu;
using RosImuPtr = sensor_msgs::msg::Imu::SharedPtr;
using RosImuConstPtr = sensor_msgs::msg::Imu::ConstSharedPtr;
using RosCloud = sensor_msgs::msg::PointCloud2;
using RosCloudConstPtr = sensor_msgs::msg::PointCloud2::ConstPtr;
using RosImage = sensor_msgs::msg::Image;
using RosImagePtr = sensor_msgs::msg::Image::SharedPtr;
using RosImageConstPtr = sensor_msgs::msg::Image::ConstSharedPtr;
using RosCompressedImage = sensor_msgs::msg::CompressedImage;
using RosCompressedImageConstPtr = sensor_msgs::msg::CompressedImage::ConstSharedPtr;
using RosPath = nav_msgs::msg::Path;
using RosPoseStamped = geometry_msgs::msg::PoseStamped;
using RosTransformStamped = geometry_msgs::msg::TransformStamped;
using RosTfQuaternion = tf2::Quaternion;

using RosImuSubscriber = rclcpp::Subscription<RosImu>::SharedPtr;
using RosCloudSubscriber = rclcpp::Subscription<RosCloud>::SharedPtr;
using RosImgSubscriber = rclcpp::Subscription<RosImage>::SharedPtr;
using RosCompressedImgSubscriber = rclcpp::Subscription<RosCompressedImage>::SharedPtr;

using RosCloudPublisher = rclcpp::Publisher<RosCloud>::SharedPtr;
using RosPathPublisher = rclcpp::Publisher<RosPath>::SharedPtr;
using RosTransformBroadcaster = std::shared_ptr<tf2_ros::TransformBroadcaster>;

#define ROS_INIT(argc, argv, name) rclcpp::init(argc, argv)
#define ROS_NODE(name) rclcpp::Node::make_shared(name)
#define ROS_SPIN(node) rclcpp::spin(node)
#define ROS_SHUTDOWN() rclcpp::shutdown()
#define ROS_OK() rclcpp::ok()
#define ROS_TIME_NOW() rclcpp::Clock().now()
#define ROS_TIMESTAMP_TO_SEC(ros_time) (rclcpp::Time(ros_time).seconds())
void SetRosTimestamp(RosHeader &header, double sec);


#define ROS_ACCESS(node, method) node->method
#define ROS_SUBSCRIBE(node, type, topic, queue, callback) node->create_subscription<type>(topic, rclcpp::QoS(queue), callback)  
#define ROS_SUBSCRIBE_MEMBER(node, type, topic, queue, callback, this) node->create_subscription<type>(topic, queue, std::bind(callback, this, std::placeholders::_1))  
#define ROS_ADVERTISE(node, type, topic, queue) node->create_publisher<type>(topic, queue)
#define ROS_GET_SUB_NUM(pub) pub->get_subscription_count()
#define ROS_PUBLISH(pub, msg) pub->publish(msg)

#define ROS_TRANSFORM_BROADCASTER(node) std::make_shared<tf2_ros::TransformBroadcaster>(node)
#define ROS_SEND_TRANSFORM(pub, msg) pub->sendTransform(msg)


#endif

#endif // ROS_COMMON_H

