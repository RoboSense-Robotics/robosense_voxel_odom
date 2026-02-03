#ifndef CAMERA_MODEL_HPP
#define CAMERA_MODEL_HPP

#include <iostream>
#include <vector>
#include <memory>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

namespace slam_cam
{

enum ProjectionModel
{
  PINHOLE = 0,
  OMNI = 1
};

enum DistortionModel
{
  RADTAN = 0,
  EQUI = 1,
  FOV = 2
};

ProjectionModel ProjectionModelConvert(const std::string &in);
std::string ProjectionModelConvert(const ProjectionModel in);
DistortionModel DistortionModelConvert(const std::string &in);
std::string DistortionModelConvert(const DistortionModel in);

class CameraBase
{
public:
  CameraBase(const int width, const int height, const ProjectionModel proj_model, const DistortionModel dist_model)
  : width_(width), height_(height), projection_model_(proj_model), distortion_model_(dist_model)
  {}
  virtual ~CameraBase() = default;
  virtual bool IsInBorder(const Eigen::Vector2d uv_dist, const int border_x=0, const int border_y=0) const
  {
    return (uv_dist[0] >= border_x && uv_dist[0] < width_ - border_x && uv_dist[1] >= border_y && uv_dist[1] < height_ - border_y);
  }
  virtual void LoadParam(std::vector<double> proj_params, std::vector<double> dist_params)
  {
    // TODO: params size check
    projection_params_ = proj_params;
    distortion_params_ = dist_params;
  }
  virtual int GetWidth() const
  {
    return width_;
  }
  virtual int GetHeight() const
  {
    return height_;
  }
  virtual std::vector<double> GetProjParam() const
  {
    return projection_params_;
  }
  virtual std::vector<double> GetDistParam() const
  {
    return distortion_params_;
  }
  virtual Eigen::Vector3d PixelToBearing(const Eigen::Vector2d &uv_dist) = 0;
  virtual Eigen::Vector2d BearingToPixel(const Eigen::Vector3d &bearing) = 0;
  virtual cv::Mat UndistortImg(cv::Mat &img_in) 
  {
    std::cout << "UndistortImg() not implemented now, proj_model: " << ProjectionModelConvert(projection_model_) << ", dist_model: " << DistortionModelConvert(distortion_model_) << std::endl; 
    std::exit(EXIT_FAILURE);
  }
  virtual void ComputePixelWeight(const double max_error, const double max_unconsistency) = 0;
  virtual cv::Mat GetUndistError() const
  {
    return undist_error_.clone();
  }
  virtual cv::Mat GetDistDelta() const
  {
    return dist_delta_.clone();
  }
  virtual cv::Mat GetDistDeltaUnconsistency() const
  {
    return dist_delta_unconsistency_.clone();
  }
  virtual cv::Mat GetPixelStd() const
  {
    return pixel_std_.clone();
  }
  virtual cv::Mat GetPixelWeight() const
  {
    return pixel_weight_.clone();
  }
  virtual double GetPixelWeight(Eigen::Vector2d uv_dist) const
  {
    if(weight_available_)
    {
      return pixel_weight_.at<float>(uv_dist[1], uv_dist[0]);
    }
    else
    {
      return 1.;
    }
  }
  virtual bool IsWeightAvailable() const
  {
    return weight_available_;
  }
protected:
  int width_{0};
  int height_{0};
  bool has_distortion_{false};
  ProjectionModel projection_model_;
  DistortionModel distortion_model_;
  std::vector<double> projection_params_;
  std::vector<double> distortion_params_;

  cv::Mat undist_error_; // CV_64DC2
  cv::Mat dist_delta_; // CV_64DC2
  cv::Mat dist_delta_unconsistency_; // CV_64DC2
  cv::Mat pixel_std_; // CV_32FC1
  cv::Mat pixel_weight_; // CV_32FC1
  bool weight_available_{false};
};

class CameraPinhole : public CameraBase
{
public:
  CameraPinhole(const int width, const int height, const ProjectionModel proj_model, const DistortionModel dist_model)
  : CameraBase(width, height, proj_model, dist_model)
  {}
  ~CameraPinhole() = default;
  void LoadParam(const std::vector<double> proj_params, const std::vector<double> dist_params) override;
  Eigen::Vector3d PixelToBearing(const Eigen::Vector2d &uv_dist) override;
  Eigen::Vector2d BearingToPixel(const Eigen::Vector3d &bearing) override;
  cv::Mat UndistortImg(cv::Mat &img_in) override;
  void ComputePixelWeight(const double max_error, const double max_unconsistency) override;
private:
  cv::Mat cvK_;
  cv::Mat undist_mapX_;
  cv::Mat undist_mapY_;
  double k1{0.}, k2{0.}, p1{0.}, p2{0.};
  double k3{0.};
  double k4{0.}, k5{0.}, k6{0.};
  double s1{0.}, s2{0.}, s3{0.}, s4{0.};
  // double tau_x{0}, tau_y{0};
};


class CameraFactor
{
public:
  static std::shared_ptr<CameraBase> CreateCamera(int width, int height, std::string proj_model, std::string dist_model)
  {
    const auto proj_model_enum = ProjectionModelConvert(proj_model);
    const auto dist_model_enum = DistortionModelConvert(dist_model);
    if(PINHOLE == proj_model_enum && RADTAN == dist_model_enum)
    {
      return std::make_shared<CameraPinhole>(width, height, proj_model_enum, dist_model_enum);
    }
    std::cout << "Unsupport camera model now, proj_model: " << proj_model << ", dist_model: " << dist_model << std::endl;
    std::exit(EXIT_FAILURE);
  }
private:
  CameraFactor() = default;
};

} // namespace slam_cam

#endif // CAMERA_MODEL_HPP