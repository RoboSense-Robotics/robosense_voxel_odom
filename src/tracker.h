#ifndef TRACKER_H
#define TRACKER_H

#include <atomic>
#include <mutex>
#include "camera_model.h"
#include "tracking_manager.h"

namespace slam_cam
{

struct CameraData
{
  double timestamp;
  std::vector<cv::Mat> bgrs;
  std::vector<cv::Mat> grays;
  std::vector<cv::Mat> masks;
};

enum ImgEnhancementMethod
{
  NONE = 0,
  HISTOGRAM = 1,
  CLAHE = 2
};

ImgEnhancementMethod ImgEnhancementMethodConvert(const std::string &in);

std::string ImgEnhancementMethodConvert(const ImgEnhancementMethod in);

class TrackerBase
{
public:
  TrackerBase(std::shared_ptr<TrackingManager> tracking_manager, std::shared_ptr<CameraBase> camera_ptr, int num_feats, std::string enhance_method) 
  : tracking_manager_(tracking_manager), camera_ptr_(camera_ptr), num_features_(num_feats), tracking_id_(1), 
  img_enhancement_method_(ImgEnhancementMethodConvert(enhance_method))
  { }
  virtual ~TrackerBase() = default;
  virtual void TrackNewFrame(const CameraData &cam_data) = 0;
  void DisplayHistory(cv::Mat &img_out, const int r1, const int g1, const int b1, const int r2, const int g2, const int b2, const int border_x, const int border_y);
protected:
  std::shared_ptr<CameraBase> camera_ptr_;
  int num_features_;
  ImgEnhancementMethod img_enhancement_method_;

  std::shared_ptr<TrackingManager> tracking_manager_;

  double timestamp_prev_; // just for vis and debug
  cv::Mat bgr_prev_;
  cv::Mat gray_prev_;
  cv::Mat mask_prev_;
  std::vector<cv::KeyPoint> kpts_prev_;
  std::vector<size_t> ids_prev_;
  std::mutex mtx_prev_;
  
  std::atomic<size_t> tracking_id_;
};

class TrackerKLT : public TrackerBase
{
public:
  TrackerKLT(std::shared_ptr<TrackingManager> tracking_manager, std::shared_ptr<CameraBase> camera, int num_feats, std::string enhance_method,
  int fast_threshold, int grid_x, int grid_y, int min_px_dist)
  : TrackerBase(tracking_manager, camera, num_feats, enhance_method),
  fast_threshold_(fast_threshold), grid_x_(grid_x), grid_y_(grid_y), min_px_dist_(min_px_dist)
  {}
  ~TrackerKLT() override = default;
  void TrackNewFrame(const CameraData &cam_data) override;
private:
  void TrackMonocularFrame(const CameraData &cam_data);
  void FeatureDetectMonocular(const std::vector<cv::Mat> &gray_pyr, const cv::Mat &mask, std::vector<cv::KeyPoint> &kpts, std::vector<size_t> &ids);
  void FeatureMatching(const std::vector<cv::Mat> &gray0_pyr, const std::vector<cv::Mat> &gray1_pyr, std::vector<cv::KeyPoint> &kpts0, std::vector<cv::KeyPoint> &kpts1, std::vector<uchar> &mask_out);
  // Parameters for our FAST grid detector
  int fast_threshold_;
  int grid_x_;
  int grid_y_;

  // Minimum pixel distance to be "far away enough" to be a different extracted feature
  int min_px_dist_;

  // How many pyramid levels to track
  int pyr_levels_ = 5;
  cv::Size win_size_ = cv::Size(15, 15);

  // Last set of image pyramids
  std::vector<cv::Mat> gray_pyramid_prev_;

  cv::Mat gray_curr_;
  std::vector<cv::Mat> gray_pyramid_curr_;
  std::mutex mtx_curr_; 

};

} // namespace slam_cam
#endif // TRACKER_H