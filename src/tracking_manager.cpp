#include "tracking_manager.h"
#include "voxel_map.hpp"
#include <iostream>

namespace slam_cam
{
int Tracking::landmark_freeze_num_{-1};

void Tracking::RemoveOlderMeasurements(const double timestamp)
{
  assert(timestamps_.size() == uvs_.size());
  assert(timestamps_.size() == bearings_.size());

  auto it1 = timestamps_.begin();
  auto it2 = uvs_.begin();
  auto it3 = bearings_.begin();

  while (it1 != timestamps_.end()) 
  {
    if (*it1 <= timestamp) 
    {
      it1 = timestamps_.erase(it1);
      it2 = uvs_.erase(it2);
      it3 = bearings_.erase(it3);
    }
    else 
    {
      break;
    }
  }
}
void Tracking::UpdateLandmarkByOdom(const int idx, const double weight, const Eigen::Vector3d &hit_pw, OctoTree *hit_octo)
{
  landmark_timestamps_.emplace_back(timestamps_[idx]);
  landmark_uvs_.emplace_back(uvs_[idx]);
  landmark_bearings_.emplace_back(bearings_[idx]);
  landmark_pws_.emplace_back(hit_pw);
  landmark_octoptrs_.emplace_back(hit_octo);
  landmark_weights_.emplace_back(weight);
  if(!freeze_flag_ && !landmark_pws_.empty())
  {
    if(landmark_pws_.size() == 1)
    {
      landmark_pw_ = landmark_pws_[0];
      has_landmark_ = true;
      landmark_nums_ = 1;
    }
    else
    {
      const int N = landmark_octoptrs_.size();
      bool ok_ptr{true};
      bool ok_p3d{false};
      Eigen::Vector3d v = Eigen::Vector3d::Zero();
      Eigen::Matrix3d P = Eigen::Matrix3d::Zero();
      double weight_sum{0.};
      for(int i=1; i<N; i++)
      {
        if(landmark_octoptrs_[i] != landmark_octoptrs_[0])
        {
          ok_ptr = false;
          break;
        }
      }
      for(int i=0; i<N; i++)
      {
        v += landmark_weights_[i] * landmark_pws_[i];
        P += landmark_weights_[i] * landmark_pws_[i] * landmark_pws_[i].transpose();
        weight_sum += landmark_weights_[i];
      }
      Eigen::Vector3d center = v/weight_sum;
      Eigen::Matrix3d cov = P/weight_sum - center*center.transpose();
      double std = std::sqrt(cov(0,0)+cov(1,1)+cov(2,2));
      if(std < landmark_octoptrs_[0]->quater_length)
      {
        ok_p3d = true;
      }
      if(ok_ptr && ok_p3d)
      {
        landmark_pw_ = center;
        has_landmark_ = true;
        landmark_nums_ = N;
        if(landmark_freeze_num_>0 && landmark_nums_>=landmark_freeze_num_)
        {
          freeze_flag_ = true;
        }
      }
      else
      {
        has_landmark_ = false;
        landmark_nums_ = 0;
      }
    }
  }
}
bool Tracking::GetVisualFactorByOdom(const std::vector<double> &timestamps, 
    Eigen::Vector3d &pw, 
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> &bearings,
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> &uvs,
    double &coeff)
{
  bool has_obs{false};
  if(has_landmark_)
  {
    pw = landmark_pw_;
    coeff = landmark_nums_;
    bearings.clear(); uvs.clear();
    bearings.resize(timestamps.size(), Eigen::Vector3d::Zero());
    uvs.resize(timestamps.size(), Eigen::Vector2d::Zero());
    for(int i=0, j=0; i<timestamps.size(); i++)
    {
      for(; j<landmark_timestamps_.size(); j++)
      {
        if(timestamps[i] == landmark_timestamps_[j])
        {
          bearings[i] = landmark_bearings_[j];
          uvs[i] = landmark_uvs_[j];
          has_obs = true;
          j++;
          break;
        }
        if(timestamps[i] < landmark_timestamps_[j])
        {
          break;
        }
      }
    }
  }
  else
  {
    pw.setZero();
    coeff = -1.;
    bearings.resize(timestamps.size(), Eigen::Vector3d::Zero());
  }
  return has_obs;
}
void Tracking::UpdateLandmarkByLocalBA(const int idx, const double weight, const Eigen::Vector3d &hit_pw, OctoTree *hit_octo)
{
  landmark_timestamps_mid_.emplace_back(timestamps_[idx]);
  landmark_uvs_mid_.emplace_back(uvs_[idx]);
  landmark_bearings_mid_.emplace_back(bearings_[idx]);
  landmark_pws_mid_.emplace_back(hit_pw);
  landmark_octoptrs_mid_.emplace_back(hit_octo);
  landmark_weights_mid_.emplace_back(weight);
  if(!freeze_flag_mid_ && !landmark_pws_mid_.empty())
  {
    if(landmark_pws_mid_.size() == 1)
    {
      landmark_pw_mid_ = landmark_pws_mid_[0];
      has_landmark_mid_ = true;
      landmark_nums_mid_ = 1;
    }
    else
    {
      const int N = landmark_octoptrs_mid_.size();
      bool ok_ptr{true};
      bool ok_p3d{false};
      Eigen::Vector3d v = Eigen::Vector3d::Zero();
      Eigen::Matrix3d P = Eigen::Matrix3d::Zero();
      double weight_sum{0.};
      for(int i=1; i<N; i++)
      {
        if(landmark_octoptrs_mid_[i] != landmark_octoptrs_mid_[0])
        {
          ok_ptr = false;
          break;
        }
      }
      for(int i=0; i<N; i++)
      {
        v += landmark_weights_mid_[i] * landmark_pws_mid_[i];
        P += landmark_weights_mid_[i] * landmark_pws_mid_[i] * landmark_pws_mid_[i].transpose();
        weight_sum += landmark_weights_mid_[i];
      }
      Eigen::Vector3d center = v/weight_sum;
      Eigen::Matrix3d cov = P/weight_sum - center*center.transpose();
      double std = std::sqrt(cov(0,0)+cov(1,1)+cov(2,2));
      if(std < landmark_octoptrs_mid_[0]->quater_length)
      {
        ok_p3d = true;
      }
      if(ok_ptr && ok_p3d)
      {
        landmark_pw_mid_ = center;
        has_landmark_mid_ = true;
        landmark_nums_mid_ = N;
        if(landmark_freeze_num_>0 && landmark_nums_mid_>=landmark_freeze_num_)
        {
          freeze_flag_mid_ = true;
        }
      }
      else
      {
        has_landmark_mid_ = false;
        landmark_nums_mid_ = 0;
      }
    }
  }
}
bool Tracking::GetVisualFactorByLocalBA(const std::vector<double> &timestamps, 
    Eigen::Vector3d &pw, 
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> &bearings,
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> &uvs,
    double &coeff)
{
  bool has_obs{false};
  if(has_landmark_)
  {
    pw = landmark_pw_;
    coeff = landmark_nums_;
    bearings.clear(); uvs.clear();
    bearings.resize(timestamps.size(), Eigen::Vector3d::Zero());
    uvs.resize(timestamps.size(), Eigen::Vector2d::Zero());
    for(int i=0, j=0; i<timestamps.size(); i++)
    {
      for(; j<landmark_timestamps_mid_.size(); j++)
      {
        if(timestamps[i] == landmark_timestamps_mid_[j])
        {
          bearings[i] = landmark_bearings_mid_[j];
          uvs[i] = landmark_uvs_mid_[j];
          has_obs = true;
          j++;
          break;
        }
        if(timestamps[i] < landmark_timestamps_mid_[j])
        {
          break;
        }
      }
    }
  }
  else
  {
    pw.setZero();
    coeff = -1.;
    bearings.resize(timestamps.size(), Eigen::Vector3d::Zero());
  }
  return has_obs;
}

void TrackingManager::AddNewTracking(size_t id, double timestamp, Eigen::Vector2d uv_dist, Eigen::Vector3d bearing)
{
  std::lock_guard<std::mutex> lck_trackings(mtx_trackings_);
  if (id_tracking_map_.find(id) != id_tracking_map_.end()) 
  {
    std::shared_ptr<Tracking> tracking = id_tracking_map_.at(id);
    tracking->uvs_.push_back(uv_dist);
    tracking->bearings_.push_back(bearing);
    tracking->timestamps_.push_back(timestamp);
    return;
  }
  std::shared_ptr<Tracking> tracking = std::make_shared<Tracking>();
  tracking->id_ = id;
  tracking->uvs_.push_back(uv_dist);
  tracking->bearings_.push_back(bearing);
  tracking->timestamps_.push_back(timestamp);
  id_tracking_map_[id] = tracking;
}

void TrackingManager::SetTrackFrame(double timestamp, std::vector<size_t> &ids)
{
  std::lock_guard<std::mutex> lck_ids(mtx_ids_);
  time_ids_map_[timestamp] = ids;
}

bool TrackingManager::GetTrackingClone(size_t id, Tracking &tracking)
{
  std::lock_guard<std::mutex> lck_trackings(mtx_trackings_);
  if(id_tracking_map_.find(id)==id_tracking_map_.end())
  {
    return false;
  }
  tracking = *id_tracking_map_.at(id);
  return true;
}

bool TrackingManager::GetTrackingPtrs(double timestamp, std::vector<std::shared_ptr<Tracking>> &tracking_ptrs)
{
  tracking_ptrs.clear();
  std::lock_guard<std::mutex> lck_trackings(mtx_trackings_);
  if(time_ids_map_.find(timestamp)!=time_ids_map_.end())
  {
    std::vector<size_t> ids = time_ids_map_.at(timestamp);
    tracking_ptrs.reserve(ids.size());
    for(size_t id:ids)
    {
      if(id_tracking_map_.find(id)!=id_tracking_map_.end())
      {
        tracking_ptrs.emplace_back(id_tracking_map_.at(id));
      }
    }
    return true;
  }
  return false;
}

void TrackingManager::RemoveOlderMeasurements(const double timestamp)
{
  {
    std::lock_guard<std::mutex> lck_trackings(mtx_trackings_);
    for(auto it = id_tracking_map_.begin(); it != id_tracking_map_.end();) 
    {
      it->second->RemoveOlderMeasurements(timestamp);
      if (it->second->timestamps_.empty()) 
      {
        id_tracking_map_.erase(it++);
      } 
      else 
      {
        it++;
      }
    }
  }
  {
    std::lock_guard<std::mutex> lck_ids(mtx_ids_);
    while(!time_ids_map_.empty())
    {
      if(time_ids_map_.begin()->first <= timestamp)
      {
        time_ids_map_.erase(time_ids_map_.begin());
      }
      else
      {
        break;
      }
    }
  }
  {
    std::lock_guard<std::mutex> lck_poses(mtx_poses_);
    while(!time_pose_cam_to_world_map_.empty())
    {
      if(time_pose_cam_to_world_map_.begin()->first <= timestamp)
      {
        time_pose_cam_to_world_map_.erase(time_pose_cam_to_world_map_.begin());
      }
      else
      {
        break;
      }
    }
  }
}

void TrackingManager::AddNewOdomPose(const double timestamp, Eigen::Matrix3d rot_cam_to_world, Eigen::Vector3d pos_cam_in_world)
{
  std::lock_guard<std::mutex> lck_poses(mtx_poses_);
  time_pose_cam_to_world_map_[timestamp] = std::make_pair(rot_cam_to_world, pos_cam_in_world);
}

void TrackingManager::AddNewLocalBAPose(const double timestamp, Eigen::Matrix3d rot_cam_to_world, Eigen::Vector3d pos_cam_in_world)
{
  std::lock_guard<std::mutex> lck_poses(mtx_poses_mid_);
  time_pose_cam_to_world_mid_map_[timestamp] = std::make_pair(rot_cam_to_world, pos_cam_in_world);
}

void TrackingManager::GetSize(std::vector<int> &lengths)
{
  lengths.clear();
  lengths.emplace_back(id_tracking_map_.size());
  lengths.emplace_back(time_ids_map_.size());
  lengths.emplace_back(time_pose_cam_to_world_map_.size());
  lengths.emplace_back(time_pose_cam_to_world_mid_map_.size());
}

} // namespace slam_cam