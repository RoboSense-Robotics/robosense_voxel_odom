#include "tracker.h"
#include "grider.hpp"

namespace slam_cam
{

ImgEnhancementMethod ImgEnhancementMethodConvert(const std::string &in)
{
  if("none" == in)
  {
    return ImgEnhancementMethod::NONE;
  }
  else if("histogram" == in)
  {
    return ImgEnhancementMethod::HISTOGRAM;
  }
  else if("clahe" == in)
  {
    return ImgEnhancementMethod::CLAHE;
  }
  else
  {
    std::cerr << "Unknown img enhancement method(str): " << in << std::endl;
    return ImgEnhancementMethod::NONE;
  }
}

std::string ImgEnhancementMethodConvert(const ImgEnhancementMethod in)
{
  switch(in)
  {
    case ImgEnhancementMethod::NONE:
      return "none";
    case ImgEnhancementMethod::HISTOGRAM:
      return "histogram";
    case ImgEnhancementMethod::CLAHE:
      return "clahe";
    default:
      std::cerr << "Unknown img enhancement method(enum): " << in << std::endl;
      return "none";
  }
}


void TrackerBase::DisplayHistory(cv::Mat &img_out, const int r1, const int g1, const int b1, const int r2, const int g2, const int b2, const int border_x, const int border_y)
{
  std::lock_guard<std::mutex> lck_prev(mtx_prev_);
  img_out = cv::Mat();
  if(!bgr_prev_.empty())
  {
    const size_t maxtracks = 33;
    img_out = bgr_prev_.clone();
    bool is_small = (std::min(img_out.cols, img_out.rows) < 400);
    const size_t kpt_num = kpts_prev_.size();
    for(int i=0; i<kpt_num; i++)
    {
      size_t id = ids_prev_[i];
      Tracking tracking;
      if(!tracking_manager_->GetTrackingClone(id, tracking))
      {
        continue;
      }
      int track_length = tracking.uvs_.size();
      bool is_out_border = true;
      if(track_length>0)
      {
        is_out_border = !camera_ptr_->IsInBorder(tracking.uvs_.at(track_length-1), border_x, border_y);
      }
      for(int j=track_length-1; j>=0; j--)
      {
        if(track_length - j > maxtracks)
          break;
        int color_r = (is_out_border ? b2 : r2) - (int)(1.0 * (is_out_border ? b1 : r1) / track_length * j);
        int color_g = (is_out_border ? r2 : g2) - (int)(1.0 * (is_out_border ? r1 : g1) / track_length * j);
        int color_b = (is_out_border ? g2 : b2) - (int)(1.0 * (is_out_border ? g1 : b1) / track_length * j);
        cv::Point2f pt_c(tracking.uvs_.at(j)(0), tracking.uvs_.at(j)(1));
        cv::circle(img_out, pt_c, (is_small) ? 1 : 2, cv::Scalar(color_r, color_g, color_b), cv::FILLED);
        // If there is a next point, then display the line from this point to the next
        if (j + 1 < track_length) 
        {
          cv::Point2f pt_n(tracking.uvs_.at(j + 1)(0), tracking.uvs_.at(j + 1)(1));
          cv::line(img_out, pt_c, pt_n, cv::Scalar(color_r, color_g, color_b));
        }
        // If the first point, display the ID
        if (j == track_length - 1) 
        {
          cv::putText(img_out, std::to_string(id), pt_c, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1, cv::LINE_AA); 
          cv::circle(img_out, pt_c, 2, cv::Scalar(0,0,255), cv::FILLED);
        }
      }
    }
    auto txtpt = (is_small) ? cv::Point(10, 30) : cv::Point(30, 60);
    cv::putText(img_out, std::to_string(timestamp_prev_), txtpt, cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0), 1, cv::LINE_AA); 
  }
}

void TrackerKLT::TrackNewFrame(const CameraData &cam_data)
{
  assert(!cam_data.grays.empty());
  assert(cam_data.grays.size()==cam_data.masks.size());

  cv::Mat gray;
  if (img_enhancement_method_ == ImgEnhancementMethod::HISTOGRAM) 
  {
    cv::equalizeHist(cam_data.grays[0], gray);
  } 
  else if (img_enhancement_method_ == ImgEnhancementMethod::CLAHE) 
  {
    double eq_clip_limit = 10.0;
    cv::Size eq_win_size = cv::Size(8, 8);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(eq_clip_limit, eq_win_size);
    clahe->apply(cam_data.grays[0], gray);
  } 
  else 
  {
    gray = cam_data.grays[0];
  }

  // Extract image pyramid
  std::vector<cv::Mat> gray_pyr;
  cv::buildOpticalFlowPyramid(gray, gray_pyr, win_size_, pyr_levels_);

  {
    std::lock_guard<std::mutex> lck_curr(mtx_curr_);
    gray_curr_ = gray;
    gray_pyramid_curr_ = gray_pyr;
  }

  TrackMonocularFrame(cam_data);
}

void TrackerKLT::TrackMonocularFrame(const CameraData &cam_data)
{
  std::lock_guard<std::mutex> lck_curr(mtx_curr_);
  cv::Mat bgr = cam_data.bgrs[0];
  cv::Mat gray = gray_curr_;
  std::vector<cv::Mat> gray_pyr = gray_pyramid_curr_;
  cv::Mat mask = cam_data.masks[0];

  if (kpts_prev_.empty()) 
  {
    // Detect new features
    std::vector<cv::KeyPoint> good_kpts_left;
    std::vector<size_t> good_ids_left;
    FeatureDetectMonocular(gray_pyr, mask, good_kpts_left, good_ids_left);
    // Save the current image and pyramid
    std::lock_guard<std::mutex> lck_prev(mtx_prev_);
    timestamp_prev_ = cam_data.timestamp;
    bgr_prev_ = bgr;
    gray_prev_ = gray;
    gray_pyramid_prev_ = gray_pyr;
    mask_prev_ = mask;
    kpts_prev_ = good_kpts_left;
    ids_prev_ = good_ids_left;
    return;
  }

  // First we should make that the last images have enough features so we can do KLT
  // This will "top-off" our number of tracks so always have a constant number
  int pts_before_detect = (int)kpts_prev_.size();
  auto pts_left_old = kpts_prev_;
  auto ids_left_old = ids_prev_;
  FeatureDetectMonocular(gray_pyramid_prev_, mask_prev_, pts_left_old, ids_left_old);

  // Our return success masks, and predicted new features
  std::vector<uchar> mask_ll;
  std::vector<cv::KeyPoint> pts_left_new = pts_left_old;

  // Lets track temporally
  FeatureMatching(gray_pyramid_prev_, gray_pyr, pts_left_old, pts_left_new, mask_ll);
  assert(pts_left_new.size() == ids_left_old.size());

  // If any of our mask is empty, that means we didn't have enough to do ransac, so just return
  if (mask_ll.empty()) {
    std::lock_guard<std::mutex> lck_prev(mtx_prev_);
    timestamp_prev_ = cam_data.timestamp;
    bgr_prev_ = bgr;
    gray_prev_ = gray;
    gray_pyramid_prev_ = gray_pyr;
    mask_prev_ = mask;
    kpts_prev_.clear();
    ids_prev_.clear();
    std::cout << "[KLT-EXTRACTOR]: Failed to get enough points to do RANSAC, resetting.....\n";
    return;
  }
  // Get our "good tracks"
  std::vector<cv::KeyPoint> good_kpts_left;
  std::vector<size_t> good_ids_left;

  // Loop through all left points
  for (size_t i = 0; i < pts_left_new.size(); i++) 
  {
    // Ensure we do not have any bad KLT tracks (i.e., points are negative)
    if (pts_left_new.at(i).pt.x < 0 || pts_left_new.at(i).pt.y < 0 || (int)pts_left_new.at(i).pt.x >= gray.cols ||
        (int)pts_left_new.at(i).pt.y >= gray.rows)
      continue;
    // Check if it is in the mask
    // NOTE: mask has max value of 255 (white) if it should be
    if ((int)cam_data.masks[0].at<uint8_t>((int)pts_left_new.at(i).pt.y, (int)pts_left_new.at(i).pt.x) > 127)
      continue;
    if (mask_ll[i]) 
    {
      good_kpts_left.push_back(pts_left_new[i]);
      good_ids_left.push_back(ids_left_old[i]);
    }
  }

  // Update our feature database, with theses new observations
  for (size_t i = 0; i < good_kpts_left.size(); i++) 
  {
    Eigen::Vector2d uv(good_kpts_left.at(i).pt.x, good_kpts_left.at(i).pt.y);
    Eigen::Vector3d bearing = camera_ptr_->PixelToBearing(uv);
    int id = good_ids_left.at(i);
    tracking_manager_->AddNewTracking(id, cam_data.timestamp, uv, bearing);
  }
  tracking_manager_->SetTrackFrame(cam_data.timestamp, good_ids_left);

  // Move forward in time
  {
    std::lock_guard<std::mutex> lck_prev(mtx_prev_);
    timestamp_prev_ = cam_data.timestamp;
    bgr_prev_ = bgr;
    gray_prev_ = gray;
    gray_pyramid_prev_ = gray_pyr;
    mask_prev_ = mask;
    kpts_prev_ = good_kpts_left;
    ids_prev_ = good_ids_left;
  }
}

void TrackerKLT::FeatureDetectMonocular(const std::vector<cv::Mat> &img_pyr, const cv::Mat &mask, std::vector<cv::KeyPoint> &kpts, std::vector<size_t> &ids)
{
  // Create a 2D occupancy grid for this current image
  // Note that we scale this down, so that each grid point is equal to a set of pixels
  // This means that we will reject points that less than grid_px_size points away then existing features
  cv::Size size_close((int)((float)img_pyr.at(0).cols / (float)min_px_dist_),
                      (int)((float)img_pyr.at(0).rows / (float)min_px_dist_)); // width x height
  cv::Mat grid_2d_close = cv::Mat::zeros(size_close, CV_8UC1);
  float size_x = (float)img_pyr.at(0).cols / (float)grid_x_;
  float size_y = (float)img_pyr.at(0).rows / (float)grid_y_;
  cv::Size size_grid(grid_x_, grid_y_); // width x height
  cv::Mat grid_2d_grid = cv::Mat::zeros(size_grid, CV_8UC1);
  cv::Mat mask0_updated = mask.clone();
  auto it0 = kpts.begin();
  auto it1 = ids.begin();
  while (it0 != kpts.end()) {
    // Get current left keypoint, check that it is in bounds
    cv::KeyPoint kpt = *it0;
    int x = (int)kpt.pt.x;
    int y = (int)kpt.pt.y;
    int edge = 10;
    if (x < edge || x >= img_pyr.at(0).cols - edge || y < edge || y >= img_pyr.at(0).rows - edge) {
      it0 = kpts.erase(it0);
      it1 = ids.erase(it1);
      continue;
    }
    // Calculate mask coordinates for close points
    int x_close = (int)(kpt.pt.x / (float)min_px_dist_);
    int y_close = (int)(kpt.pt.y / (float)min_px_dist_);
    if (x_close < 0 || x_close >= size_close.width || y_close < 0 || y_close >= size_close.height) {
      it0 = kpts.erase(it0);
      it1 = ids.erase(it1);
      continue;
    }
    // Calculate what grid cell this feature is in
    int x_grid = std::floor(kpt.pt.x / size_x);
    int y_grid = std::floor(kpt.pt.y / size_y);
    if (x_grid < 0 || x_grid >= size_grid.width || y_grid < 0 || y_grid >= size_grid.height) {
      it0 = kpts.erase(it0);
      it1 = ids.erase(it1);
      continue;
    }
    // Check if this keypoint is near another point
    if (grid_2d_close.at<uint8_t>(y_close, x_close) > 127) {
      it0 = kpts.erase(it0);
      it1 = ids.erase(it1);
      continue;
    }
    // Now check if it is in a mask area or not
    // NOTE: mask has max value of 255 (white) if it should be
    if (mask.at<uint8_t>(y, x) > 127) {
      it0 = kpts.erase(it0);
      it1 = ids.erase(it1);
      continue;
    }
    // Else we are good, move forward to the next point
    grid_2d_close.at<uint8_t>(y_close, x_close) = 255;
    if (grid_2d_grid.at<uint8_t>(y_grid, x_grid) < 255) {
      grid_2d_grid.at<uint8_t>(y_grid, x_grid) += 1;
    }
    // Append this to the local mask of the image
    if (x - min_px_dist_ >= 0 && x + min_px_dist_ < img_pyr.at(0).cols && y - min_px_dist_ >= 0 && y + min_px_dist_ < img_pyr.at(0).rows) {
      cv::Point pt1(x - min_px_dist_, y - min_px_dist_);
      cv::Point pt2(x + min_px_dist_, y + min_px_dist_);
      cv::rectangle(mask0_updated, pt1, pt2, cv::Scalar(255), -1);
    }
    it0++;
    it1++;
  }

  // First compute how many more features we need to extract from this image
  // If we don't need any features, just return
  double min_feat_percent = 0.50;
  int num_featsneeded = num_features_ - (int)kpts.size();
  if (num_featsneeded < std::min(20, (int)(min_feat_percent * num_features_)))
    return;

  // This is old extraction code that would extract from the whole image
  // This can be slow as this will recompute extractions for grid areas that we have max features already
  // std::vector<cv::KeyPoint> pts0_ext;
  // Grider_FAST::perform_griding(img_pyr.at(0), mask0_updated, pts0_ext, num_features_, grid_x_, grid_y_, fast_threshold_, true);

  // We also check a downsampled mask such that we don't extract in areas where it is all masked!
  cv::Mat mask0_grid;
  cv::resize(mask, mask0_grid, size_grid, 0.0, 0.0, cv::INTER_NEAREST);

  // Create grids we need to extract from and then extract our features (use fast with griding)
  int num_features_grid = (int)((double)num_features_ / (double)(grid_x_ * grid_y_)) + 1;
  int num_features_grid_req = std::max(1, (int)(min_feat_percent * num_features_grid));
  std::vector<std::pair<int, int>> valid_locs;
  for (int x = 0; x < grid_2d_grid.cols; x++) {
    for (int y = 0; y < grid_2d_grid.rows; y++) {
      if ((int)grid_2d_grid.at<uint8_t>(y, x) < num_features_grid_req && (int)mask0_grid.at<uint8_t>(y, x) != 255) {
        valid_locs.emplace_back(x, y);
      }
    }
  }
  std::vector<cv::KeyPoint> pts0_ext;
  GriderGrid::perform_griding(img_pyr.at(0), mask0_updated, valid_locs, pts0_ext, num_features_, grid_x_, grid_y_, fast_threshold_, true);

  // Now, reject features that are close a current feature
  std::vector<cv::KeyPoint> kpts0_new;
  std::vector<cv::Point2f> pts0_new;
  for (auto &kpt : pts0_ext) {
    // Check that it is in bounds
    int x_grid = (int)(kpt.pt.x / (float)min_px_dist_);
    int y_grid = (int)(kpt.pt.y / (float)min_px_dist_);
    if (x_grid < 0 || x_grid >= size_close.width || y_grid < 0 || y_grid >= size_close.height)
      continue;
    // See if there is a point at this location
    if (grid_2d_close.at<uint8_t>(y_grid, x_grid) > 127)
      continue;
    // Else lets add it!
    kpts0_new.push_back(kpt);
    pts0_new.push_back(kpt.pt);
    grid_2d_close.at<uint8_t>(y_grid, x_grid) = 255;
  }

  // Loop through and record only ones that are valid
  for (size_t i = 0; i < pts0_new.size(); i++) {
    // update the uv coordinates
    kpts0_new.at(i).pt = pts0_new.at(i);
    // append the new uv coordinate
    kpts.push_back(kpts0_new.at(i));
    // move id foward and append this new point
    size_t temp = ++tracking_id_;
    ids.push_back(temp);
  }
}

void TrackerKLT::FeatureMatching(const std::vector<cv::Mat> &gray0_pyr, const std::vector<cv::Mat> &gray1_pyr, std::vector<cv::KeyPoint> &kpts0, std::vector<cv::KeyPoint> &kpts1, std::vector<uchar> &mask_out)
{
  // We must have equal vectors
  assert(kpts0.size() == kpts1.size());

  // Return if we don't have any points
  if (kpts0.empty() || kpts1.empty())
    return;

  // Convert keypoints into points (stupid opencv stuff)
  std::vector<cv::Point2f> pts0, pts1;
  for (size_t i = 0; i < kpts0.size(); i++) {
    pts0.push_back(kpts0.at(i).pt);
    pts1.push_back(kpts1.at(i).pt);
  }

  // If we don't have enough points for ransac just return empty
  // We set the mask to be all zeros since all points failed RANSAC
  if (pts0.size() < 10) {
    for (size_t i = 0; i < pts0.size(); i++)
      mask_out.push_back((uchar)0);
    return;
  }

  // Now do KLT tracking to get the valid new points
  std::vector<uchar> mask_klt;
  std::vector<float> error;
  cv::TermCriteria term_crit = cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
  cv::calcOpticalFlowPyrLK(gray0_pyr, gray1_pyr, pts0, pts1, mask_klt, error, win_size_, pyr_levels_, term_crit, cv::OPTFLOW_USE_INITIAL_FLOW);

  // Normalize these points, so we can then do ransac
  // We don't want to do ransac on distorted image uvs since the mapping is nonlinear
  std::vector<cv::Point2f> pts0_n, pts1_n;
  for (size_t i = 0; i < pts0.size(); i++) 
  {
    Eigen::Vector2d uv0(pts0.at(i).x, pts0.at(i).y);
    Eigen::Vector3d bearing0 = camera_ptr_->PixelToBearing(uv0);
    Eigen::Vector2d uv1(pts1.at(i).x, pts1.at(i).y);
    Eigen::Vector3d bearing1 = camera_ptr_->PixelToBearing(uv1);
    bearing0.head(2) /= bearing0[2];
    bearing1.head(2) /= bearing1[2];
    pts0_n.push_back(cv::Point2f(bearing0[0], bearing0[1]));
    pts1_n.push_back(cv::Point2f(bearing1[0], bearing1[1]));
    auto proj_params = camera_ptr_->GetProjParam();
  }

  // Do RANSAC outlier rejection (note since we normalized the max pixel error is now in the normalized cords)
  std::vector<uchar> mask_rsc;
  auto proj_params = camera_ptr_->GetProjParam();
  double max_focallength = std::max(proj_params[0], proj_params[1]);
  cv::findFundamentalMat(pts0_n, pts1_n, cv::FM_RANSAC, 2.0 / max_focallength, 0.999, mask_rsc);

  // Loop through and record only ones that are valid
  for (size_t i = 0; i < mask_klt.size(); i++) 
  {
    auto mask = (uchar)((i < mask_klt.size() && mask_klt[i] && i < mask_rsc.size() && mask_rsc[i]) ? 1 : 0);
    mask_out.push_back(mask);
  }

  // Copy back the updated positions
  for (size_t i = 0; i < pts0.size(); i++) 
  {
    kpts0.at(i).pt = pts0.at(i);
    kpts1.at(i).pt = pts1.at(i);
  }
}

} // namespace slam_cam