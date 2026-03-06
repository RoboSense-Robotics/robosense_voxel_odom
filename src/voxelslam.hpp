#pragma once

#include "ros_common.h"
#include "globals.h"
#include "tools.hpp"
#include "ekf_imu.hpp"
#include "voxel_map.hpp"
#include "feature_point.hpp"
#include <mutex>
#include <Eigen/Eigenvalues>
#include <malloc.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <malloc.h>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>

#include "tracker.h"
#include <random> 
#include <yaml-cpp/yaml.h>

using namespace std;

RosCloudPublisher pub_scan, pub_cmap, pub_init, pub_pmap;
RosCloudPublisher pub_prev_path, pub_curr_path;
RosPathPublisher pub_trajectory;
RosCloudPublisher pub_scan_rgb;
RosImuSubscriber sub_imu;
RosCloudSubscriber sub_pcl;
RosImgSubscriber sub_cam;
RosCompressedImgSubscriber sub_cam_compressed;
RosPath path;

template <typename T>
inline bool YamlRead(const YAML::Node& yaml, const std::string& key, T& out_val)
{
  if (!yaml[key] || yaml[key].Type() == YAML::NodeType::Null)
  {
    std::cout << "key:" << key << " doesn't exist." << std::endl;
    exit(1);
    return false;
  }
  else
  {
    out_val = yaml[key].as<T>();
    return true;
  }
}

inline bool YamlNodeKeyExist(const YAML::Node& yaml, const std::string& key)
{
  return yaml[key].IsDefined();
}

template <typename T>
void pub_pl_func(T &pl, RosCloudPublisher &pub)
{
  pl.height = 1; pl.width = pl.size();
  RosCloud output;
  pcl::toROSMsg(pl, output);
  output.header.frame_id = "camera_init";
  output.header.stamp = ROS_TIME_NOW();
  ROS_PUBLISH(pub, output);
}

mutex mBuf;
Features feat;
deque<RosImuPtr> imu_buf;
deque<pcl::PointCloud<PointType>::Ptr> pcl_buf;
deque<double> time_buf;
std::deque<double> img_time_buf;
std::deque<cv::Mat> img_buf;
std::shared_ptr<slam_cam::CameraBase> camera_ptr;
std::shared_ptr<slam_cam::TrackerBase> tracker_ptr;
std::shared_ptr<slam_cam::TrackingManager> tracking_manager_ptr;
double img_scale{1.};
double weight_scale_unit{1.};
double reprojection_cov{1.};
int vis_update_status{0};
const bool use_landmark{true};
bool re_weight{false};
int border_x{0};
int border_y{0};
std::vector<double> plane_density;
double scale_xy_{1.};
double scale_z_{1.};

double imu_last_time = -1;
int point_notime = 0;
double last_pcl_time = -1;
std::atomic<double> last_img_time{-1.};
bool ray_tracking_all_map{true};
std::array<double, 2> ray_tracking_range{0.1, 30.};

bool landmark_use_dist_weight{false};
bool vio_use_dist_weight{false};
bool vba_use_dist_weight{false};

std::ofstream time_cost_total_;

int sync_mode{0};
bool img_enable{true};
int img_ds_ratio{1};

void imu_handler(const RosImuConstPtr &msg_in)
{
  static int flag = 1;
  if(flag)
  {
    flag = 0;
    printf("Time0: %lf\n", ROS_TIMESTAMP_TO_SEC(msg_in->header.stamp));
  }

  RosImuPtr msg(new RosImu(*msg_in));

  mBuf.lock();
  imu_last_time = ROS_TIMESTAMP_TO_SEC(msg->header.stamp);
  imu_buf.push_back(msg);
  mBuf.unlock();
}

void pcl_handler(const RosCloudConstPtr &msg)
{
  auto start = std::chrono::high_resolution_clock::now();
  pcl::PointCloud<PointType>::Ptr pl_ptr(new pcl::PointCloud<PointType>());
  double t0 = feat.process(msg, *pl_ptr);
  if(pl_ptr->empty())
  {
    PointType ap; 
    ap.x = 0; ap.y = 0; ap.z = 0; 
    ap.intensity = 0; ap.curvature = 0;
    pl_ptr->push_back(ap);
    ap.curvature = 0.09;
    pl_ptr->push_back(ap);
  }

  sort(pl_ptr->begin(), pl_ptr->end(), [](PointType &x, PointType &y)
  {
    return x.curvature < y.curvature;
  });
  while(pl_ptr->back().curvature > 0.11)
    pl_ptr->points.pop_back();

  mBuf.lock();
  time_buf.push_back(t0);
  pcl_buf.push_back(pl_ptr);
  mBuf.unlock();

  auto end = std::chrono::high_resolution_clock::now();
  time_cost_total_ << "pcl_handler," << (std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()*1e-3)
          << "," << std::to_string(std::chrono::duration<double>(start.time_since_epoch()).count())
          << "," << std::to_string(std::chrono::duration<double>(end.time_since_epoch()).count())
          << ",,"
          << std::to_string(t0)
          << std::endl;
}

void monocular_handler(const RosImageConstPtr &msg) 
{
  if(!img_enable) return;
  static int cnt = -1;
  cnt++;
  auto start = std::chrono::high_resolution_clock::now();
  slam_cam::CameraData cam_data;
  cam_data.timestamp = ROS_TIMESTAMP_TO_SEC(msg->header.stamp);
  cv::Mat bgr = cv_bridge::toCvShare(msg, "bgr8")->image;
  cv::resize(bgr, bgr, cv::Size(), img_scale, img_scale, cv::INTER_LINEAR);
  
  if(0 == sync_mode || 0 == cnt % img_ds_ratio)
  {
    mBuf.lock();
    img_time_buf.push_back(cam_data.timestamp);
    img_buf.push_back(bgr);
    mBuf.unlock();
  }

  cv::Mat gray;
  cv::cvtColor(bgr, gray, CV_BGR2GRAY);
  cam_data.bgrs.emplace_back(bgr);
  cam_data.grays.emplace_back(gray);
  cam_data.masks.emplace_back(cv::Mat::zeros(gray.rows, gray.cols, CV_8UC1));
  tracker_ptr->TrackNewFrame(cam_data);
  last_img_time = cam_data.timestamp;
  auto end = std::chrono::high_resolution_clock::now();
  time_cost_total_ << "monocular_handler," << (std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()*1e-3)
          << "," << std::to_string(std::chrono::duration<double>(start.time_since_epoch()).count())
          << "," << std::to_string(std::chrono::duration<double>(end.time_since_epoch()).count())
          << ",,"
          << std::to_string(cam_data.timestamp)
          << std::endl;
}

void monocular_compressed_handler(const RosCompressedImageConstPtr &msg) 
{
  if(!img_enable) return;
  static int cnt = -1;
  cnt++;
  auto start = std::chrono::high_resolution_clock::now();
  slam_cam::CameraData cam_data;
  cam_data.timestamp = ROS_TIMESTAMP_TO_SEC(msg->header.stamp);
  cv::Mat bgr = cv_bridge::toCvCopy(msg, "bgr8")->image;
  cv::resize(bgr, bgr, cv::Size(), img_scale, img_scale, cv::INTER_LINEAR);
  
  if(0 == sync_mode || 0 == cnt % img_ds_ratio)
  {
    mBuf.lock();
    img_time_buf.push_back(cam_data.timestamp);
    img_buf.push_back(bgr);
    mBuf.unlock();
  }

  cv::Mat gray;
  cv::cvtColor(bgr, gray, CV_BGR2GRAY);
  cam_data.bgrs.emplace_back(bgr);
  cam_data.grays.emplace_back(gray);
  cam_data.masks.emplace_back(cv::Mat::zeros(gray.rows, gray.cols, CV_8UC1));
  tracker_ptr->TrackNewFrame(cam_data);
  last_img_time = cam_data.timestamp;
  auto end = std::chrono::high_resolution_clock::now();
  time_cost_total_ << "monocular_handler," << (std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()*1e-3)
          << "," << std::to_string(std::chrono::duration<double>(start.time_since_epoch()).count())
          << "," << std::to_string(std::chrono::duration<double>(end.time_since_epoch()).count())
          << ",,"
          << std::to_string(cam_data.timestamp)
          << std::endl;
}

bool sync_packages_new(pcl::PointCloud<PointType>::Ptr &pl_ptr, std::deque<RosImuPtr> &imus, IMUEKF &p_imu)
{
  static bool pl_ready = false;
  static bool img_ready = false;
  if(img_enable)
  {
    if(!img_ready)
    {
      if(img_time_buf.empty()) return false;
      if(last_pcl_time < 0)
      {
        mBuf.lock();
        last_pcl_time = img_time_buf.front();
        img_time_buf.pop_front();
        img_buf.pop_front();
        mBuf.unlock();
        return false;
      }
      p_imu.pcl_beg_time = last_pcl_time;
      mBuf.lock();
      last_pcl_time = img_time_buf.front();
      p_imu.img_match = img_buf.front();
      img_time_buf.pop_front();
      img_buf.pop_front();
      mBuf.unlock();
      p_imu.pcl_end_time = last_pcl_time;
      p_imu.img_match_time = last_pcl_time;
      p_imu.img_match_status = 1;

      std::cout << "p_imu.pcl_beg_time:" << std::to_string(p_imu.pcl_beg_time) << ", p_imu.pcl_end_time:" << std::to_string(p_imu.pcl_end_time) << std::endl;
      img_ready = true;
    }
    if(!pl_ready)
    {
      if(pcl_buf.empty()) return false;
      mBuf.lock();
      double point_begin_time = time_buf[0];
      double point_end_time = time_buf.back() + pcl_buf.back()->back().curvature;
      mBuf.unlock();
      if(point_begin_time >= p_imu.pcl_end_time)
      {
        img_ready = false;
        last_pcl_time = p_imu.pcl_beg_time;
        std::cout << "Reset img_ready due to mismatch time range!!!"
        << " p_imu.pcl_beg_time:" << std::to_string(p_imu.pcl_beg_time) << ", p_imu.pcl_end_time:" << std::to_string(p_imu.pcl_end_time)
        << ", point_begin_time:" << std::to_string(point_begin_time) << ", point_end_time:" << std::to_string(point_end_time) 
        << std::endl;
        return false;
      }
      if(point_end_time < p_imu.pcl_end_time)
      {
        return false;
      }
      mBuf.lock();

      std::cout << "Match img-lidar ok!!!"
        << " p_imu.pcl_beg_time:" << std::to_string(p_imu.pcl_beg_time) << ", p_imu.pcl_end_time:" << std::to_string(p_imu.pcl_end_time)
        << ", point_begin_time:" << std::to_string(point_begin_time) << ", point_end_time:" << std::to_string(point_end_time) 
        << std::endl;
      pl_ptr->clear();
      int drop_num{0};
      for(int i=0; i<time_buf.size(); i++)
      {
        pcl::PointCloud<PointType>::Ptr cloud_cur = pcl_buf[i];
        double time_cur = time_buf[i];
        int j=0;
        for(; j<cloud_cur->size(); j++)
        {
          double time_point = time_cur + cloud_cur->points[j].curvature;
          if(time_point <= p_imu.pcl_beg_time)
          {
            continue;
          }
          else if(time_point <= p_imu.pcl_end_time)
          {
            pl_ptr->points.push_back(cloud_cur->points[j]);
            pl_ptr->points.back().curvature = time_point - p_imu.pcl_beg_time;
          }
          else
          {
            break;
          }
        }
        if(cloud_cur->size() == j)
        {
          drop_num++;
          continue;
        }
        else
        {
          break;
        }
      }
      std::cout << "drop_num:" << drop_num << std::endl;
      if(pl_ptr->size()<100)
      {
        img_ready = false;
        last_pcl_time = p_imu.pcl_beg_time;
        std::cout << "Reset img_ready due to too few points!!!"
        << " p_imu.pcl_beg_time:" << std::to_string(p_imu.pcl_beg_time) << ", p_imu.pcl_end_time:" << std::to_string(p_imu.pcl_end_time)
        << ", point_begin_time:" << std::to_string(point_begin_time) << ", point_end_time:" << std::to_string(point_end_time) 
        << std::endl;
        mBuf.unlock();
        return false;
      }
      while(drop_num>0)
      {
        pcl_buf.pop_front(); 
        time_buf.pop_front();
        drop_num--;
      }
      pl_ready = true;
      mBuf.unlock();
    }
  }
  else
  {
    if(!pl_ready)
    {
      if(pcl_buf.empty()) return false;

      mBuf.lock();
      pl_ptr = pcl_buf.front();
      p_imu.pcl_beg_time = time_buf.front();
      pcl_buf.pop_front(); time_buf.pop_front();
      mBuf.unlock();

      p_imu.pcl_end_time = p_imu.pcl_beg_time + pl_ptr->back().curvature;
      p_imu.img_match_status = 0;

      if(point_notime)
      {
        if(last_pcl_time < 0)
        {
          last_pcl_time = p_imu.pcl_beg_time;
          return false;
        }

        p_imu.pcl_end_time = p_imu.pcl_beg_time;
        p_imu.pcl_beg_time = last_pcl_time;
        last_pcl_time = p_imu.pcl_end_time;
      }
      std::cout << "p_imu.pcl_beg_time:" << std::to_string(p_imu.pcl_beg_time) << ", p_imu.pcl_end_time:" << std::to_string(p_imu.pcl_end_time) << std::endl;

      p_imu.img_match_status = -2;
      p_imu.img_match = cv::Mat();
      pl_ready = true;
    }
  }

  if(!pl_ready || imu_last_time <= p_imu.pcl_end_time) return false;
  mBuf.lock();
  double imu_time = ROS_TIMESTAMP_TO_SEC(imu_buf.front()->header.stamp);
  while((!imu_buf.empty()) && (imu_time < p_imu.pcl_end_time)) 
  {
    imu_time = ROS_TIMESTAMP_TO_SEC(imu_buf.front()->header.stamp);
    if(imu_time > p_imu.pcl_end_time) break;
    imus.push_back(imu_buf.front());
    imu_buf.pop_front();
  }
  mBuf.unlock();

  if(imu_buf.empty())
  {
    printf("imu buf empty\n"); exit(0);
  }

  pl_ready = false;
  img_ready = false;

  if(imus.size() > 4)
    return true;
  else
    return false;
}

bool sync_packages(pcl::PointCloud<PointType>::Ptr &pl_ptr, std::deque<RosImuPtr> &imus, IMUEKF &p_imu)
{
  static bool pl_ready = false;

  if(!pl_ready)
  {
    if(pcl_buf.empty()) return false;

    mBuf.lock();
    pl_ptr = pcl_buf.front();
    p_imu.pcl_beg_time = time_buf.front();
    pcl_buf.pop_front(); time_buf.pop_front();
    mBuf.unlock();

    p_imu.pcl_end_time = p_imu.pcl_beg_time + pl_ptr->back().curvature;
    p_imu.img_match_status = 0;

    if(point_notime)
    {
      if(last_pcl_time < 0)
      {
        last_pcl_time = p_imu.pcl_beg_time;
        return false;
      }

      p_imu.pcl_end_time = p_imu.pcl_beg_time;
      p_imu.pcl_beg_time = last_pcl_time;
      last_pcl_time = p_imu.pcl_end_time;
    }

    pl_ready = true;
  }

  if(!pl_ready || imu_last_time <= p_imu.pcl_end_time) return false;
  if(img_enable && img_time_buf.size()>1 && 0==p_imu.img_match_status)
  {
    p_imu.img_match_status = -1;
    mBuf.lock();
    for(int i=img_time_buf.size()-2; i>=0; i--)
    {
      if(img_time_buf[i]<p_imu.pcl_end_time)
      {
        double dt = img_time_buf[i+1] - p_imu.pcl_end_time;
        if(dt>=0)
        {
          if(dt<35e-3)
          {
            p_imu.img_match_status = 1;
            std::cout << "[1] dt:" << dt
              << ", img_match_time:" << std::to_string(img_time_buf[i+1]) 
              << ", pcl_end_time:" << std::to_string(p_imu.pcl_end_time) 
              << std::endl;
            p_imu.img_match_time = img_time_buf[i+1];
            p_imu.img_match = img_buf[i+1];
            p_imu.pcl_end_time = p_imu.img_match_time;
          }
          else
          {
            p_imu.img_match_status = -1;
            p_imu.img_match = cv::Mat();
            std::cout << "[-1] dt:" << dt
              << ", img_match_time:" << std::to_string(img_time_buf[i+1]) 
              << ", pcl_end_time:" << std::to_string(p_imu.pcl_end_time) 
              << std::endl;
          }
        }
        else
        {
          p_imu.img_match_status = 0;
        }
        break;
      }
    }  
    mBuf.unlock();
    return false;
  }
  if(0==p_imu.img_match_status && img_time_buf.size()<=1)
  {
    p_imu.img_match_status = -2;
    p_imu.img_match = cv::Mat();
    std::cout << "[-2] fail to match..." << std::endl;
  }

  mBuf.lock();
  double imu_time = ROS_TIMESTAMP_TO_SEC(imu_buf.front()->header.stamp);
  while((!imu_buf.empty()) && (imu_time < p_imu.pcl_end_time)) 
  {
    imu_time = ROS_TIMESTAMP_TO_SEC(imu_buf.front()->header.stamp);
    if(imu_time > p_imu.pcl_end_time) break;
    imus.push_back(imu_buf.front());
    imu_buf.pop_front();
  }
  mBuf.unlock();

  if(imu_buf.empty())
  {
    printf("imu buf empty\n"); exit(0);
  }

  pl_ready = false;

  if(imus.size() > 4)
    return true;
  else
    return false;
}

double dept_err, beam_err;
void calcBodyVar(Eigen::Vector3d &pb, const float range_inc, const float degree_inc, Eigen::Matrix3d &var) 
{
  if (pb[2] == 0)
    pb[2] = 0.0001;
  float range = sqrt(pb[0] * pb[0] + pb[1] * pb[1] + pb[2] * pb[2]);
  float range_var = range_inc * range_inc;
  Eigen::Matrix2d direction_var;
  direction_var << pow(sin(DEG2RAD(degree_inc)), 2), 0, 0, pow(sin(DEG2RAD(degree_inc)), 2);
  Eigen::Vector3d direction(pb);
  direction.normalize();
  Eigen::Matrix3d direction_hat;
  direction_hat << 0, -direction(2), direction(1), direction(2), 0, -direction(0), -direction(1), direction(0), 0;
  Eigen::Vector3d base_vector1(1, 1, -(direction(0) + direction(1)) / direction(2));
  base_vector1.normalize();
  Eigen::Vector3d base_vector2 = base_vector1.cross(direction);
  base_vector2.normalize();
  Eigen::Matrix<double, 3, 2> N;
  N << base_vector1(0), base_vector2(0), base_vector1(1), base_vector2(1), base_vector1(2), base_vector2(2);
  Eigen::Matrix<double, 3, 2> A = range * direction_hat * N;
  var = direction * range_var * direction.transpose() + A * direction_var * A.transpose();
};

// Compute the variance of the each point
void var_init(IMUST &ext, pcl::PointCloud<PointType> &pl_cur, PVecPtr pptr, double dept_err, double beam_err)
{
  int plsize = pl_cur.size();
  pptr->clear();
  pptr->resize(plsize);
  for(int i=0; i<plsize; i++)
  {
    PointType &ap = pl_cur[i];
    pointVar &pv = pptr->at(i);
    pv.pnt << ap.x, ap.y, ap.z;
    calcBodyVar(pv.pnt, dept_err, beam_err, pv.var);
    pv.pnt = ext.R * pv.pnt + ext.p;
    pv.var = ext.R * pv.var * ext.R.transpose();
    pv.intensity = ap.intensity;
  }
}

void var_init(IMUST &ext, pcl::PointCloud<PointType> &pl_cur, PVecPtr pptr)
{
  int plsize = pl_cur.size();
  pptr->clear();
  pptr->resize(plsize);
  for(int i=0; i<plsize; i++)
  {
    PointType &ap = pl_cur[i];
    pointVar &pv = pptr->at(i);
    pv.pnt << ap.x, ap.y, ap.z;
    pv.pnt = ext.R * pv.pnt + ext.p;
    pv.intensity = ap.intensity;
  }
}

void pvec_update(PVecPtr pptr, IMUST &x_curr, PLV(3) &pwld)
{
  Eigen::Matrix3d rot_var = x_curr.cov.block<3, 3>(0, 0);
  Eigen::Matrix3d tsl_var = x_curr.cov.block<3, 3>(3, 3);

  for(pointVar &pv: *pptr)
  {
    Eigen::Matrix3d phat = hat(pv.pnt);
    pv.var = x_curr.R * pv.var * x_curr.R.transpose() + phat * rot_var * phat.transpose() + tsl_var;
    pwld.push_back(x_curr.R * pv.pnt + x_curr.p);
  }
}

double get_memory()
{
  ifstream infile("/proc/self/status");
  double mem = -1;
  string lineStr, str;
  while(getline(infile, lineStr))
  {
    stringstream ss(lineStr);
    bool is_find = false;
    while(ss >> str)
    {
      if(str == "VmRSS:")
      {
        is_find = true; continue;
      }

      if(is_find) mem = stod(str);
      break;
    }
    if(is_find) break;
  }
  return mem / (1024); // [MB] //mem / (1048576); // [GB]
}


bool RayTracking(std::unordered_map<VOXEL_LOC, OctoTree*> &map, 
  Eigen::Vector3d orig, Eigen::Vector3d dir, double voxel_size, double min_range, double max_range, const bool ray_tracking_all_map, Eigen::Vector3d &hit_pt, OctoTree* &hit_octo)
{
  const double step = voxel_size*0.5;
  VOXEL_LOC last_position(0,0,0);
  double hit_density{0.};
  for(double range=std::max(min_range, 0.01); range<=max_range; range+=step)
  {
    Eigen::Vector3d p3d = orig + dir*range;
    float loc_xyz[3];
    for(int j=0; j<3; j++)
    {
      loc_xyz[j] = p3d[j] / voxel_size;
      if(loc_xyz[j] < 0)
        loc_xyz[j] -= 1.0;
    }
    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    if(last_position == position)
    {
      continue;
    }
    last_position = position;
    auto iter = map.find(position);
    if(iter == map.end())
    {
      continue;
    }
    else
    {
      Eigen::Vector3d p3d_inW;
      OctoTree *octo = nullptr;
      double dist{-1.};
      iter->second->RayTracking(orig, dir, min_range, max_range, ray_tracking_all_map, p3d_inW, octo, dist);
      if(nullptr != octo && dist<2.*2.)
      {
        double S = std::sqrt(std::abs(octo->plane.eig_value(1))) * std::sqrt(std::abs(octo->plane.eig_value(2)));
        double density = double(octo->last_num) / S;
        if(density > 1.5*hit_density)
        {
          hit_density = density;
          hit_pt = p3d_inW;
          hit_octo = octo;
          if(hit_density > plane_density.back())
          {
            return true;
          }
        }
      }
    }
  }
  if(hit_octo != nullptr)
  {
    if(hit_density > plane_density[0])
    {
      return true;
    }
    else
    {
      hit_octo = nullptr;
      hit_pt.setZero();
    }
  }
  return false;
}

void UpdateLandmarkByOdom(std::unordered_map<VOXEL_LOC, OctoTree*> &map, 
  std::vector<std::shared_ptr<slam_cam::Tracking>> &tracking_ptrs,
  double timestamp, Eigen::Matrix3d rot_cam_to_world, Eigen::Vector3d pos_cam_in_world,
  double voxel_size, double min_range_ref, double max_range_ref, const bool ray_tracking_all_map,
  cv::Mat range_img, const int radius)
{
  bool adaptive_range = (min_range_ref<0.) || (max_range_ref<0.);
  std::cout << "adaptive_range:" << adaptive_range << std::endl;
  Eigen::Vector3d orig = pos_cam_in_world;
  for(auto &tracking_ptr: tracking_ptrs)
  {
    int idx_landmark{-1};
    for(int i=tracking_ptr->timestamps_.size()-1; i>=0; i--)
    {
      if(timestamp == tracking_ptr->timestamps_[i])
      {
        idx_landmark = i;
        break;
      }
      if(timestamp > tracking_ptr->timestamps_[i])
      {
        break;
      }
    }
    if(idx_landmark >= 0)
    {
      Eigen::Vector3d dir = rot_cam_to_world * tracking_ptr->bearings_[idx_landmark];
      double weight = landmark_use_dist_weight ? camera_ptr->GetPixelWeight(tracking_ptr->uvs_[idx_landmark]) : 1.; 
      Eigen::Vector3d hit_pw{Eigen::Vector3d::Zero()};
      OctoTree* hit_octo(nullptr);
      if(adaptive_range)
      {
        Eigen::Vector2d uv_dist = tracking_ptr->uvs_[idx_landmark];
        float min_range = 10000.f;
        float max_range = 0.f;
        for(int v=uv_dist[1]-radius; v<=uv_dist[1]+radius; v++)
        {
          for(int u=uv_dist[0]-radius; u<=uv_dist[0]+radius; u++)
          {
            if(v<0 || v>=range_img.rows || u<0 || u>=range_img.cols) continue;
            float range = range_img.at<float>(v, u);
            if(range<0.01f) continue;
            min_range = std::min(min_range, range);
            max_range = std::max(max_range, range);
          }
        }
        if(min_range <= max_range)
        {
          min_range -= voxel_size;
          max_range += voxel_size;
          min_range = min_range_ref<0. ? min_range : min_range_ref;
          max_range = max_range_ref<0. ? max_range : max_range_ref;
          if(weight>0. && RayTracking(map, orig, dir, voxel_size, min_range, max_range, ray_tracking_all_map, hit_pw, hit_octo))
          {
            tracking_ptr->UpdateLandmarkByOdom(idx_landmark, weight, hit_pw, hit_octo);
          }
        }
      }
      else
      {
        if(weight>0. && RayTracking(map, orig, dir, voxel_size, min_range_ref, max_range_ref, ray_tracking_all_map, hit_pw, hit_octo))
        {
          tracking_ptr->UpdateLandmarkByOdom(idx_landmark, weight, hit_pw, hit_octo);
        }
      }
    }
  }
}

void UpdateLandmarkByLocalBA(std::unordered_map<VOXEL_LOC, OctoTree*> &map, 
  std::vector<std::shared_ptr<slam_cam::Tracking>> &tracking_ptrs,
  double timestamp, Eigen::Matrix3d rot_cam_to_world, Eigen::Vector3d pos_cam_in_world,
  double voxel_size, double min_range_ref, double max_range_ref, const bool ray_tracking_all_map,
  cv::Mat range_img, const int radius)
{
  bool adaptive_range = (min_range_ref<0.) || (max_range_ref<0.);
  std::cout << "adaptive_range:" << adaptive_range << std::endl;
  Eigen::Vector3d orig = pos_cam_in_world;
  for(auto &tracking_ptr: tracking_ptrs)
  {
    int idx_landmark{-1};
    for(int i=tracking_ptr->timestamps_.size()-1; i>=0; i--)
    {
      if(timestamp == tracking_ptr->timestamps_[i])
      {
        idx_landmark = i;
        break;
      }
      if(timestamp > tracking_ptr->timestamps_[i])
      {
        break;
      }
    }
    if(idx_landmark >= 0)
    {
      Eigen::Vector3d dir = rot_cam_to_world * tracking_ptr->bearings_[idx_landmark];
      double weight = landmark_use_dist_weight ? camera_ptr->GetPixelWeight(tracking_ptr->uvs_[idx_landmark]) : 1.; 
      Eigen::Vector3d hit_pw;
      OctoTree* hit_octo;
      if(adaptive_range)
      {
        Eigen::Vector2d uv_dist = tracking_ptr->uvs_[idx_landmark];
        float min_range = 10000.f;
        float max_range = 0.f;
        for(int v=uv_dist[1]-radius; v<=uv_dist[1]+radius; v++)
        {
          for(int u=uv_dist[0]-radius; u<=uv_dist[0]+radius; u++)
          {
            if(v<0 || v>=range_img.rows || u<0 || u>=range_img.cols) continue;
            float range = range_img.at<float>(v, u);
            if(range<0.01f) continue;
            min_range = std::min(min_range, range);
            max_range = std::max(max_range, range);
          }
        }
        if(min_range <= max_range)
        {
          min_range -= voxel_size;
          max_range += voxel_size;
          min_range = min_range_ref<0. ? min_range : min_range_ref;
          max_range = max_range_ref<0. ? max_range : max_range_ref;
          if(weight>0. && RayTracking(map, orig, dir, voxel_size, min_range, max_range, ray_tracking_all_map, hit_pw, hit_octo))
          {
            tracking_ptr->UpdateLandmarkByLocalBA(idx_landmark, weight, hit_pw, hit_octo);
          }
        }
      }
      else
      {
        if(weight>0. && RayTracking(map, orig, dir, voxel_size, min_range_ref, max_range_ref, ray_tracking_all_map, hit_pw, hit_octo))
        {
          tracking_ptr->UpdateLandmarkByLocalBA(idx_landmark, weight, hit_pw, hit_octo);
        }
      }
    }
  }
}

void GetVisualFactorByOdom(std::vector<std::shared_ptr<slam_cam::Tracking>> &tracking_ptrs, 
  std::vector<IMUST> &xs,VisualFactor &visual_factor)
{
  std::vector<double> timestamp_list;
  timestamp_list.reserve(xs.size());
  for(auto &x:xs)
  {
    timestamp_list.emplace_back(x.t);
  }
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> bearings;
  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> uvs;
  bearings.reserve(xs.size());
  uvs.reserve(xs.size());
  Eigen::Vector3d pw;
  double coeff;
  for(auto &tracking_ptr: tracking_ptrs)
  {
    if(tracking_ptr->GetVisualFactorByOdom(timestamp_list, pw, bearings, uvs, coeff))
    {
      visual_factor.push_feat(pw, bearings, uvs, coeff, tracking_ptr->id_);
    }
  }
}

void GetVisualFactorByLocalBA(std::vector<std::shared_ptr<slam_cam::Tracking>> &tracking_ptrs, 
  std::vector<IMUST> &xs,VisualFactor &visual_factor)
{
  std::vector<double> timestamp_list;
  timestamp_list.reserve(xs.size());
  for(auto &x:xs)
  {
    timestamp_list.emplace_back(x.t);
  }
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> bearings;
  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> uvs;
  bearings.reserve(xs.size());
  uvs.reserve(xs.size());
  Eigen::Vector3d pw;
  double coeff;
  for(auto &tracking_ptr: tracking_ptrs)
  {
    if(tracking_ptr->GetVisualFactorByLocalBA(timestamp_list, pw, bearings, uvs, coeff))
    {
      visual_factor.push_feat(pw, bearings, uvs, coeff, tracking_ptr->id_);
    }
  }
}

void GetVisualFactorByOdom(std::vector<std::shared_ptr<slam_cam::Tracking>> &tracking_ptrs, 
  const double timestamp, std::unordered_map<size_t, Eigen::Vector3d> &id_pw_map,
  std::vector<std::pair<size_t, std::pair<Eigen::Vector3d, Eigen::Vector2d>>> &id_bearing_uvs)
{
  id_pw_map.clear();
  id_bearing_uvs.clear();
  for(auto &tracking_ptr: tracking_ptrs)
  {
    if(tracking_ptr->has_landmark_)
    {
      Eigen::Vector3d pw = tracking_ptr->landmark_pw_;
      for(int j=tracking_ptr->timestamps_.size()-1; j>=0; j--)
      {
        if(timestamp == tracking_ptr->timestamps_[j])
        {
          id_pw_map[tracking_ptr->id_] = pw;
          id_bearing_uvs.emplace_back(std::make_pair(tracking_ptr->id_, std::make_pair(tracking_ptr->bearings_[j], tracking_ptr->uvs_[j])));
          break;
        }
      }
    }
  }
}