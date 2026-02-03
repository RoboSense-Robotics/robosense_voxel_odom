#ifndef TRACKING_MANAGER_H
#define TRACKING_MANAGER_H

#include <Eigen/Eigen>
#include <memory>
#include <unordered_map>
#include <map>
#include <mutex>

class OctoTree;
namespace slam_cam
{

class Tracking 
{
public:
  size_t id_{0};
  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> uvs_; // pixel coordinates in raw img
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> bearings_; // undistored and normalized bearing vectors
  std::vector<double> timestamps_;
  bool to_delete_{false};
  void RemoveOlderMeasurements(const double timestamp);

  std::vector<double> landmark_timestamps_;
  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> landmark_uvs_;
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> landmark_bearings_;
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> landmark_pws_;
  std::vector<OctoTree*> landmark_octoptrs_;
  std::vector<double> landmark_weights_;
  Eigen::Vector3d landmark_pw_{0, 0, 0};
  bool freeze_flag_{false};
  bool has_landmark_{false};
  int landmark_nums_{0};
  void UpdateLandmarkByOdom(const int idx, const double weight, const Eigen::Vector3d &hit_pw, OctoTree *hit_octo);
  bool GetVisualFactorByOdom(const std::vector<double> &timestamps, 
    Eigen::Vector3d &pw, 
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> &bearings,
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> &uvs,
    double &coeff);

  std::vector<double> landmark_timestamps_mid_;
  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> landmark_uvs_mid_;
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> landmark_bearings_mid_;
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> landmark_pws_mid_;
  std::vector<OctoTree*> landmark_octoptrs_mid_;
  std::vector<double> landmark_weights_mid_;
  Eigen::Vector3d landmark_pw_mid_{0, 0, 0};
  bool freeze_flag_mid_{false};
  bool has_landmark_mid_{false};
  int landmark_nums_mid_{0};
  void UpdateLandmarkByLocalBA(const int idx, const double weight, const Eigen::Vector3d &hit_pw, OctoTree *hit_octo);
  bool GetVisualFactorByLocalBA(const std::vector<double> &timestamps, 
    Eigen::Vector3d &pw, 
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> &bearings,
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> &uvs,
    double &coeff);
private:

public:
  static int landmark_freeze_num_;
};

struct TriangleInfo
{
public:
  size_t id{0};
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> bearings; // undistored and normalized bearing vectors
  std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>> rot_cam_to_world;
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> pos_cam_in_world;
};

class TrackingManager
{
public:
  TrackingManager() = default;
  ~TrackingManager() = default;
  void AddNewTracking(size_t id, double timestamp, Eigen::Vector2d uv_dist, Eigen::Vector3d bearing);
  bool GetTrackingClone(size_t id, Tracking &tracking);
  bool GetTrackingPtrs(double timestamp, std::vector<std::shared_ptr<Tracking>> &tracking_ptrs);
  void RemoveOlderMeasurements(const double timestamp);
  void SetTrackFrame(double timestamp, std::vector<size_t> &ids);

  void AddNewOdomPose(const double timestamp, Eigen::Matrix3d rot_cam_to_world, Eigen::Vector3d pos_cam_in_world);
  void AddNewLocalBAPose(const double timestamp, Eigen::Matrix3d rot_cam_to_world, Eigen::Vector3d pos_cam_in_world);

  void GetSize(std::vector<int> &lengths);
private:
  std::unordered_map<size_t, std::shared_ptr<Tracking>> id_tracking_map_;
  std::mutex mtx_trackings_;

  std::map<double, std::vector<size_t>> time_ids_map_;
  std::mutex mtx_ids_;

  std::map<double, std::pair<Eigen::Matrix3d, Eigen::Vector3d>> time_pose_cam_to_world_map_;
  std::mutex mtx_poses_;

  std::map<double, std::pair<Eigen::Matrix3d, Eigen::Vector3d>> time_pose_cam_to_world_mid_map_;
  std::mutex mtx_poses_mid_;
};

} // namespace slam_cam
#endif // TRACKING_MANAGER_H