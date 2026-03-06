# robosense_voxel_odom: A LiDAR-Visual-Inertial Odom System

[中文版](README_CN.md)

## 1. Introduction

**robosense_voxel_odom** is a modified version of **[Voxel-SLAM](https://github.com/hku-mars/Voxel-SLAM)** with enhanced odometry and removed backend components, adapted for the Active Camera. It can be initialized with static state and dynamic state. You can run it in LiDAR-Inertial Odometry mode or LiDAR-Visual-Inertial Odometry mode. This respository supports both ROS1 and ROS2.

## 2. Prerequisited

* Ubuntu (tested on 20.04 and 22.04)

* [ROS](http://wiki.ros.org/ROS/Installation) (tested with Noetic and Humble)

* [PCL](https://pointclouds.org/) (tested with 1.10)

* [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) (tested with 3.3.7)

* [OpenCV](https://opencv.org/‌) (tested with 4.5.0)

## 3. Build

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

## 4. Run robosense_voxel_odom
By default, all outputs from the robosense_voxel_odom are saved in ```<ros_workspace>/src/robosense_voxel_odom/Log```.
### Demo1

Download ros2 bag from: [Climbing Spot](https://cdn.robosense.cn/AC_wiki/zuopaotai.zip)

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

### Demo2

Download ros2 bag from: [European architecture](https://cdn.robosense.cn/AC_wiki/shuichi.zip)

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

### Run with your device

You need to provide the config folder of your device in `robosense_voxel_odom/config` like ac_zuopaotai folder used in Demo1.

If you can get a Active Camera, you can setup your config-folder as following:

1. Create config folder from demo1 config folder :```cd <ros_workspace>/src/robosense_voxel_odom/config && cp -r ac_zuopaotai/ ac```;
2. Replace calibration info in the `ac/calibration.yaml` with your device calibration parameters;
3. Set the `lid_topic`/`imu_topic`/`img_topic`/`compressed_img_topic` in the `ac/odom.yaml` to your topic names.

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

## 5. Acknowledgements

Thanks for the following work:

* [Voxel-SLAM](https://github.com/hku-mars/Voxel-SLAM) (Voxel-SLAM: A Complete, Accurate, and Versatile LiDAR-Inertial SLAM System)

* [OpenVINS](https://github.com/rpng/open_vins) (OpenVINS: A Research Platform for Visual-Inertial Estimation)

* [LIR-LIVO](https://github.com/IF-A-CAT/LIR-LIVO) (LIR-LIVO: A Lightweight,Robust LiDAR/Vision/Inertial Odometry with Illumination-Resilient Deep Features)

## 6. License

The source code is released under [GPLv2](https://www.gnu.org/licenses/) license.
