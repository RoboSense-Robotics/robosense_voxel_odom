# robosense_voxel_odom: A LiDAR-Visual-Inertial Odom System

[English Version](README.md)

## 1. 介绍

**robosense_voxel_odom** 是 **[Voxel-SLAM](https://github.com/hku-mars/Voxel-SLAM)** 的修改版，增强了里程计功能，同时移除了后端优化部分，并适配了Active Camera 。它既可静态初始化，也可动态初始化。您可以选择在雷达-惯性里程计模式或雷达-视觉-惯性里程计模式下运行。本仓库同时支持ROS1和ROS2。

## 2. 依赖

* Ubuntu (在 20.04 和 22.04 系统上经过测试)

* [ROS](http://wiki.ros.org/ROS/Installation) (在 Noetic and Humble 版本上经过测试)

* [PCL](https://pointclouds.org/) (在 1.10 版本上经过测试)

* [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) (在 3.3.7 版本上经过测试)

* [OpenCV](https://opencv.org/‌) (在 4.5.0 版本上经过测试)

## 3. 安装编译

### ROS1

```sh
cd <ros_workspace>/src
git clone https://github.com/RoboSense-Robotics/robosense_voxel_odom
cd ../ && catkin build -v -i
source devel/setup.bash
```

### ROS2

```sh
cd <ros_workspace>/src
git clone https://github.com/RoboSense-Robotics/robosense_voxel_odom
cd ../ && colcon build --symlink-install --event-handlers console_direct+
source install/setup.bash
```

## 4. 运行 robosense_voxel_odom
默认情况下，robosense_voxel_odom 的所有输出都将保存到 ```<ros_workspace>/src/robosense_voxel_odom/Log``` 目录下。
### 示例 1

您可以从 [Climbing Spot](https://cdn.robosense.cn/AC_wiki/zuopaotai.zip)下载示例 1 的 ROS2 数据包

```sh
# terminal 1 to run odom
cd <ros_workspace>
source install/setup.bash
ros2 run robosense_voxel_odom voxelodom ac_zuopaotai  
# terminal 2 to run rviz
cd <ros_workspace>/src/robosense_voxel_odom/rviz_cfg
rviz2 -d ac_ros2.rviz
# terminal 3 to play ros2 bag
cd <demo1’s_download_location‌>
ros2 bag play zuopaotai
```

### 示例 2

您可以从 [European architecture](https://cdn.robosense.cn/AC_wiki/shuichi.zip)下载示例 2 的 ROS2 数据包

```sh
# terminal 1 to run odom
cd <ros_workspace>
source install/setup.bash
ros2 run robosense_voxel_odom voxelodom ac_shuichi  
# terminal 2 to run rviz
cd <ros_workspace>/src/robosense_voxel_odom/rviz_cfg
rviz2 -d ac_ros2.rviz
# terminal 3 to play ros2 bag
cd <demo2’s_download_location‌>
ros2 bag play shuichi
```

### 在您的传感器数据上运行

您需要在 `robosense_voxel_odom/config` 目录下提供和您传感器相对应的配置目录，就像示例 1 中使用的 ac_zuopaotai 那样。

如果您有 Active Camera 设备，您可以按照如下步骤设置配置目录：　

1. 基于示例 1 的配置目录创建新配置目录： ```cd <ros_workspace>/src/robosense_voxel_odom/config && cp -r ac_zuopaotai/ ac```；
2. 用您设备的标定参数替换 `ac/calibration.yaml` 中的相关标定信息；
3. 将 `ac/odom.yaml` 中的参数 `lid_topic`/`imu_topic`/`img_topic`/`compressed_img_topic` 设置为您设备的话题名。

#### ROS1

```sh
# terminal 1 to run odom
cd <ros_workspace>
source devel/setup.bash
rosrun robosense_voxel_odom voxelodom ac 
# terminal 2 to run rviz
cd <ros_workspace>/src/robosense_voxel_odom/rviz_cfg
rviz -d ac.rviz
# terminal 3 to play ros1 bag
rosbag play <your_ros1_bag>
```

#### ROS2

```sh
# terminal 1 to run odom
cd <ros_workspace>
source install/setup.bash
ros2 run robosense_voxel_odom voxelodom ac  
# terminal 2 to run rviz
cd <ros_workspace>/src/robosense_voxel_odom/rviz_cfg
rviz2 -d ac_ros2.rviz
# terminal 3 to play ros2 bag
ros2 bag play <your_ros2_bag>
```

## 5. 致谢

感谢以下的工作:

* [Voxel-SLAM](https://github.com/hku-mars/Voxel-SLAM) (Voxel-SLAM: A Complete, Accurate, and Versatile LiDAR-Inertial SLAM System)

* [OpenVINS](https://github.com/rpng/open_vins) (OpenVINS: A Research Platform for Visual-Inertial Estimation)

* [LIR-LIVO](https://github.com/IF-A-CAT/LIR-LIVO) (LIR-LIVO: A Lightweight,Robust LiDAR/Vision/Inertial Odometry with Illumination-Resilient Deep Features)

## 6. License

该仓库在 [GPLv2](https://www.gnu.org/licenses/) 协议下开源。