#ifndef EKF_IMU_HPP
#define EKF_IMU_HPP

#include "ros_common.h"
#include "tools.hpp"
#include <deque>
#include <opencv2/opencv.hpp>

class IMUEKF
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  bool init_flag;
  double pcl_beg_time, pcl_end_time, last_pcl_end_time;
  double img_match_time{-1.};
  int img_match_status{0}; // 0-need to be matched; 1-matched; -1:fail to match
  cv::Mat img_match;
  int init_num;
  Eigen::Vector3d mean_acc, mean_gyr;
  RosImuPtr last_imu;
  int min_init_num = 30;
  Eigen::Vector3d angvel_last, acc_s_last;

  Eigen::Vector3d cov_acc, cov_gyr;
  Eigen::Vector3d cov_bias_gyr, cov_bias_acc;

  Eigen::Matrix3d Lid_rot_to_IMU;
  Eigen::Vector3d Lid_offset_to_IMU;

  Eigen::Matrix3d rot_cam_to_imu_;
  Eigen::Vector3d pos_cam_in_imu_;

  double scale_gravity = 1.0;
  std::vector<IMUST> imu_poses;

  int point_notime = 0;

  int init_mode{0}; // 0: default, 1: static, 2: dynamic
  bool gravity_align_en = true;
  IMUST x_first;

  IMUST x_prev;

  IMUEKF()
  {
    init_flag = false;
    init_num = 0;
    mean_acc.setZero(); mean_gyr.setZero();
    angvel_last.setZero(); acc_s_last.setZero();
    x_prev.t = -1.;
  }

  void motion_blur(IMUST &xc, pcl::PointCloud<PointType> &pcl_in, std::deque<RosImuPtr> &imus)
  {
    IMUST x_curr = xc;
    imus.push_front(last_imu);

    imu_poses.clear();
    // imu_poses.emplace_back(0, xc.R, xc.p, xc.v, angvel_last, acc_s_last);

    Eigen::Vector3d acc_imu, angvel_avr, acc_avr, vel_imu(xc.v), pos_imu(xc.p);
    Eigen::Matrix3d R_imu(xc.R);
    Eigen::Matrix<double, DIM, DIM> F_x, cov_w;

    double dt = 0;
    for(auto it_imu=imus.begin(); it_imu!=imus.end()-1; it_imu++)
    {
      RosImu &head = **(it_imu);
      RosImu &tail = **(it_imu+1);

      if(ROS_TIMESTAMP_TO_SEC(tail.header.stamp) < last_pcl_end_time) continue;

      angvel_avr << 0.5*(head.angular_velocity.x + tail.angular_velocity.x), 
                    0.5*(head.angular_velocity.y + tail.angular_velocity.y), 
                    0.5*(head.angular_velocity.z + tail.angular_velocity.z);
      acc_avr << 0.5*(head.linear_acceleration.x + tail.linear_acceleration.x), 
                 0.5*(head.linear_acceleration.y + tail.linear_acceleration.y), 
                 0.5*(head.linear_acceleration.z + tail.linear_acceleration.z);

      angvel_avr -= xc.bg;
      acc_avr = acc_avr * scale_gravity - xc.ba;
      acc_imu = R_imu * acc_avr + xc.g;

      double cur_time = ROS_TIMESTAMP_TO_SEC(head.header.stamp);
      if(cur_time < last_pcl_end_time)
        cur_time = last_pcl_end_time;
      dt = ROS_TIMESTAMP_TO_SEC(tail.header.stamp) - cur_time;

      double offt = cur_time - pcl_beg_time;
      imu_poses.emplace_back(offt, R_imu, pos_imu, vel_imu, angvel_avr, acc_imu);
      
      Eigen::Matrix3d acc_avr_skew = hat(acc_avr);
      Eigen::Matrix3d Exp_f = Exp(angvel_avr, dt);

      F_x.setIdentity();
      cov_w.setZero();

      F_x.block<3,3>(0,0)  = Exp(angvel_avr, - dt);
      F_x.block<3,3>(0,9)  = -I33 * dt;
      F_x.block<3,3>(3,6)  = I33 * dt;
      F_x.block<3,3>(6,0)  = - R_imu * acc_avr_skew * dt;
      F_x.block<3,3>(6,12) = - R_imu * dt;
      cov_w.block<3,3>(0,0).diagonal() = cov_gyr * dt * dt;
      cov_w.block<3,3>(6,6) = R_imu * cov_acc.asDiagonal() * R_imu.transpose() * dt * dt;
      cov_w.block<3,3>(9,9).diagonal()   = cov_bias_gyr * dt * dt;
      cov_w.block<3,3>(12,12).diagonal() = cov_bias_acc * dt * dt;

      xc.cov = F_x * xc.cov * F_x.transpose() + cov_w;
      pos_imu = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt;
      vel_imu = vel_imu + acc_imu * dt;
      R_imu = R_imu * Exp_f;

    }

    double imu_end_time = ROS_TIMESTAMP_TO_SEC(imus.back()->header.stamp);
    double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
    dt = note * (pcl_end_time - imu_end_time);
    xc.v = vel_imu + note * acc_imu * dt;
    xc.R = R_imu * Exp(note*angvel_avr, dt);
    xc.p = pos_imu + note * vel_imu * dt + note * 0.5 * acc_imu * dt * dt;
    xc.t = pcl_end_time;

    RosImuPtr imu1(new RosImu(*imus.front()));
    RosImuPtr imu2(new RosImu(*imus.back()));
    
    SetRosTimestamp(imu1->header, last_pcl_end_time);
    SetRosTimestamp(imu2->header, pcl_end_time);
    // imus.pop_front();
    last_imu = imus.back();
    last_pcl_end_time = pcl_end_time;
    imus.front() = imu1;
    imus.back()  = imu2;

    if(point_notime)
      return;

    auto it_pcl = pcl_in.end() - 1;
    for(int i=imu_poses.size()-1; i>=0; i--)
    {
      IMUST &head = imu_poses[i];
      R_imu = head.R;
      acc_imu = head.ba;
      vel_imu = head.v;
      pos_imu = head.p;
      angvel_avr = head.bg;

      for(; it_pcl->curvature > head.t; it_pcl--)
      {
        dt = it_pcl->curvature - head.t;

        Eigen::Matrix3d R_i = R_imu * Exp(angvel_avr, dt);
        Eigen::Vector3d T_ei = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - xc.p;

        Eigen::Vector3d P_i(it_pcl->x, it_pcl->y, it_pcl->z);
        Eigen::Vector3d P_compensate = Lid_rot_to_IMU.transpose() * (xc.R.transpose() * (R_i * (Lid_rot_to_IMU * P_i + Lid_offset_to_IMU) + T_ei) - Lid_offset_to_IMU);

        it_pcl->x = P_compensate(0);
        it_pcl->y = P_compensate(1);
        it_pcl->z = P_compensate(2);
        if(it_pcl == pcl_in.begin()) break;
      }
    }
    if(it_pcl != pcl_in.begin())
    {
      if(x_prev.t>0.)
      {
        double t_prev = x_prev.t;
        Eigen::Quaterniond q_prev(x_prev.R);
        Eigen::Vector3d p_prev(x_prev.p);
        double t_cur = x_curr.t;
        Eigen::Quaterniond q_cur(x_curr.R);
        Eigen::Vector3d p_cur(x_curr.p);
        for(; ; it_pcl--)
        {
          double time = pcl_beg_time + it_pcl->curvature;
          if(time < x_prev.t) break;
          double k = (time - t_prev) / (t_cur - t_prev);
          Eigen::Matrix3d R_i = (q_prev.slerp(k, q_cur)).toRotationMatrix();
          Eigen::Vector3d T_ei = (1.-k) * p_prev + k * p_cur - xc.p;

          Eigen::Vector3d P_i(it_pcl->x, it_pcl->y, it_pcl->z);
          Eigen::Vector3d P_compensate = Lid_rot_to_IMU.transpose() * (xc.R.transpose() * (R_i * (Lid_rot_to_IMU * P_i + Lid_offset_to_IMU) + T_ei) - Lid_offset_to_IMU);

          it_pcl->x = P_compensate(0);
          it_pcl->y = P_compensate(1);
          it_pcl->z = P_compensate(2);
          if(it_pcl == pcl_in.begin()) break;
        }
      }
    }
    x_prev = x_curr;
  }

  void IMU_init(deque<RosImuPtr> &imus)
  {
    Eigen::Vector3d cur_acc, cur_gyr;
    for(RosImuPtr imu: imus)
    {
      cur_acc << imu->linear_acceleration.x,
                 imu->linear_acceleration.y,
                 imu->linear_acceleration.z;
      cur_gyr << imu->angular_velocity.x,
                 imu->angular_velocity.y,
                 imu->angular_velocity.z;

      if(init_num != 0)
      {
        mean_acc += (cur_acc - mean_acc) / init_num; 
        mean_gyr += (cur_gyr - mean_gyr) / init_num;
      }
      else
      {
        mean_acc = cur_acc; // modify
        mean_gyr = cur_gyr;
        init_num = 1;
      }
      
      init_num++;
    }

    last_imu = imus.back();
  }

  int process(IMUST &x_curr, pcl::PointCloud<PointType> &pcl_in, std::deque<RosImuPtr> &imus)
  {
    if(!init_flag)
    {
      IMU_init(imus);
      if(1 == init_mode)
      {
        scale_gravity = G_m_s2 / mean_acc.norm();
        x_curr.bg = Eigen::Vector3d::Zero(); // mean_gyr;
        x_curr.cov.setIdentity();
        x_curr.cov *= 0.0001;
        x_curr.cov.block<6, 6>(9, 9) = Eigen::Matrix<double, 6, 6>::Identity() * 0.00001;
      }
      else
      {
        if(mean_acc.norm() < 2)
          scale_gravity = G_m_s2;
      }
      printf("scale_gravity: %lf %lf %d\n", scale_gravity, mean_acc.norm(), init_num);
      x_curr.g = -mean_acc * scale_gravity;
      if(init_num > min_init_num) init_flag = true;
      last_pcl_end_time = pcl_end_time;

      if(1==init_mode && init_flag)
      {
        Eigen::Matrix3d G_R_I0 = Eigen::Matrix3d::Identity();
        if(gravity_align_en)
        {
          Eigen::Vector3d ez(0, 0, -1), gz(x_curr.g);
          Eigen::Quaterniond G_q_I0 = Eigen::Quaterniond::FromTwoVectors(gz, ez);
          G_R_I0 = G_q_I0.toRotationMatrix();
        }

        x_curr.R = G_R_I0 * x_curr.R;
        x_curr.g = G_R_I0 * x_curr.g;
        for(int i=0; i<3; i++)
        {
          if(std::abs(x_curr.g[i])<1e-4)
          {
            x_curr.g[i] = 0.;
          }
        }

        std::cout << "Align pcl_end_time:" << std::to_string(pcl_end_time) 
        << ", x_curr.g:" << x_curr.g.transpose()
        << ", R:\n" << x_curr.R 
        << std::endl;
        
        x_first = x_curr;
        std::cout << "init cov:" << x_curr.cov.diagonal().transpose() << std::endl;
        std::cout << "cov_gyr:" << cov_gyr.transpose() << std::endl;
        std::cout << "cov_acc:" << cov_acc.transpose() << std::endl;
        std::cout << "cov_bias_gyr:" << cov_bias_gyr.transpose() << std::endl;
        std::cout << "cov_bias_acc:" << cov_bias_acc.transpose() << std::endl;

        std::cout 
          << "p:" << x_curr.p.transpose()
          << ", v:" << x_curr.v.transpose()
          << ", bg:" << x_curr.bg.transpose()
          << ", ba:" << x_curr.ba.transpose()
          << ", g:" << x_curr.g.transpose()
          << ", R:\n" << x_curr.R
          << std::endl;
      }

      return 0;
    }

    motion_blur(x_curr, pcl_in, imus);
    return 1;
  }

};

#endif

