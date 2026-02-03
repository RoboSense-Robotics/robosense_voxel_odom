#include "ros_common.h"

#ifdef ROS_VERSION_1
void SetRosTimestamp(RosHeader &header, double sec)
{
	header.stamp = ros::Time(sec);
}

#elif defined(ROS_VERSION_2)

void SetRosTimestamp(RosHeader &header, double sec)
{
	int64_t total_nsec = static_cast<int64_t>(sec * 1e9);
	header.stamp.sec = static_cast<int32_t>(total_nsec / 1000000000LL);
	header.stamp.nanosec = static_cast<uint32_t>(total_nsec % 1000000000LL);
}

#endif