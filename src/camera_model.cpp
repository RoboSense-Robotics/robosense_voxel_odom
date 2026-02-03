#include "camera_model.h"

namespace slam_cam
{

ProjectionModel ProjectionModelConvert(const std::string &in)
{
  if("pinhole" == in)
  {
    return ProjectionModel::PINHOLE;
  }
  else if("omni" == in)
  {
    return ProjectionModel::OMNI;
  }
  else
  {
    std::cerr << "Unknown projection model(str): " << in << std::endl;
    std::exit(EXIT_FAILURE);
  }
}
std::string ProjectionModelConvert(const ProjectionModel in)
{
  switch(in)
  {
    case ProjectionModel::PINHOLE:
      return "pinhole";
    case ProjectionModel::OMNI:
      return "omni";
    default:
      std::cerr << "Unknown projection model(enum): " << in << std::endl;
      std::exit(EXIT_FAILURE);
  }
}
DistortionModel DistortionModelConvert(const std::string &in)
{
  if("radtan" == in)
  {
    return DistortionModel::RADTAN;
  }
  else if("equi" == in)
  {
    return DistortionModel::EQUI;
  }
  else if("fov" == in)
  {
    return DistortionModel::FOV;
  }
  else
  {
    std::cerr << "Unknown distortion model(str): " << in << std::endl;
    std::exit(EXIT_FAILURE);
  }
}
std::string DistortionModelConvert(const DistortionModel in)
{
  switch(in)
  {
    case DistortionModel::RADTAN:
      return "radtan";
    case DistortionModel::EQUI:
      return "equi";
    case DistortionModel::FOV:
      return "fov";
    default:
      std::cerr << "Unknown distortion model(enum): " << in << std::endl;
      std::exit(EXIT_FAILURE);
  }
}

void CameraPinhole::LoadParam(const std::vector<double> proj_params, const std::vector<double> dist_params)
{
  assert(proj_params.size() == 4);
  assert(dist_params.size()==0 || dist_params.size()==4 || dist_params.size()==5 || dist_params.size()==8 || dist_params.size()==12); //|| dist_params.size() ==14);
  projection_params_ = proj_params;
  distortion_params_ = dist_params;
  cvK_ = (cv::Mat_<double>(3, 3) << proj_params[0], 0, proj_params[2], 0, proj_params[1], proj_params[3], 0, 0, 1);
  cv::initUndistortRectifyMap(cvK_, distortion_params_, std::vector<double>(), cvK_,
                            cv::Size(width_, height_), CV_16SC2, undist_mapX_, undist_mapY_);
  has_distortion_ = false;
  for(auto &d:distortion_params_)
  {
    if(d != 0)
    {
      has_distortion_ = true;
      break;
    }
  }
  if(has_distortion_)
  {
    k1 = distortion_params_[0];
    k2 = distortion_params_[1];
    p1 = distortion_params_[2];
    p2 = distortion_params_[3];
    if(distortion_params_.size()>=5)
    {
      k3 = distortion_params_[4];
    }
    if(distortion_params_.size()>=8)
    {
      k4 = distortion_params_[5];
      k5 = distortion_params_[6];
      k6 = distortion_params_[7];
    }
    if(distortion_params_.size()>=12)
    {
      s1 = distortion_params_[8];
      s2 = distortion_params_[9];
      s3 = distortion_params_[10];
      s4 = distortion_params_[11];
    }
    // if(distortion_params_.size()>=14)
    // {
    //   tau_x = distortion_params_[12];
    //   tau_y = distortion_params_[13];
    // }
  }
  else
  {
    k1 = k2 = p1 = p2 = 0.;
    k3 = 0.;
    k4 = k5 = k6 = 0.;
    s1 = s2 = s3 = s4 = 0.;
    // tau_x = tau_y = 0.;
  }
}
Eigen::Vector3d CameraPinhole::PixelToBearing(const Eigen::Vector2d &uv_dist)
{
  Eigen::Vector3d bearing;
  if(has_distortion_)
  {
    cv::Mat pt(1, 2, CV_32F);
    pt.at<float>(0, 0) = uv_dist[0];
    pt.at<float>(0, 1) = uv_dist[1];
    pt = pt.reshape(2);
    cv::undistortPoints(pt, pt, cvK_, distortion_params_);
    bearing[0] = pt.at<cv::Point2f>(0, 0).x;
    bearing[1] = pt.at<cv::Point2f>(0, 0).y;
    bearing[2] = 1.0;
  }
  else
  {
    bearing[0] = (uv_dist[0] - projection_params_[2]) / projection_params_[0];
    bearing[1] = (uv_dist[1] - projection_params_[3]) / projection_params_[1];
    bearing[2] = 1.0;
  }
  return bearing.normalized();
}
Eigen::Vector2d CameraPinhole::BearingToPixel(const Eigen::Vector3d &bearing)
{
  Eigen::Vector2d uv_dist;
  Eigen::Vector2d xy_unitz = bearing.head(2) / bearing[2]; // TODO: check z=0
  if(has_distortion_)
  {
    const double x = xy_unitz[0];
    const double y = xy_unitz[1];
    const double x2 = x * x;
    const double xy = x * y;
    const double y2 = y * y;
    const double r2 = x2 + y2;
    const double r4 = r2 * r2;
    const double r6 = r4 * r2;

    const double k = (1. + k1*r2 + k2* r4 + k3*r6)/(1. + k4*r2 + k5*r4 + k6*r6);

    double x1 = x*k + 2.*p1*xy + p2*(r2 + 2.*x2) + s1*r2 + s2*r4;
    double y1 = y*k + p1*(r2 + 2.*y2) + 2.*p2*xy + s3*r2 + s4*r4;

    uv_dist[0] = projection_params_[0] * x1 + projection_params_[2];
    uv_dist[1] = projection_params_[1] * y1 + projection_params_[3];
  }
  else
  {
    uv_dist[0] = projection_params_[0] * xy_unitz[0] + projection_params_[2];
    uv_dist[1] = projection_params_[1] * xy_unitz[1] + projection_params_[3];
  }
  return uv_dist;
}
cv::Mat CameraPinhole::UndistortImg(cv::Mat &img_in)
{
  cv::Mat img_out;
  if(has_distortion_)
    cv::remap(img_in, img_out, undist_mapX_, undist_mapY_, cv::INTER_LINEAR);
  else
    img_out = img_in.clone();
  return img_out;
}
void CameraPinhole::ComputePixelWeight(const double max_error, const double max_unconsistency)
{
  weight_available_ = false;
  undist_error_ = cv::Mat(height_, width_, CV_64FC2, cv::Vec2d(0,0));
  dist_delta_ = cv::Mat(height_, width_, CV_64FC2, cv::Vec2d(0,0));
  dist_delta_unconsistency_ = cv::Mat(height_, width_, CV_64FC2, cv::Vec2d(0,0));
  pixel_std_ = cv::Mat(height_, width_, CV_32FC1, cv::Scalar(0.f));
  pixel_weight_ = cv::Mat(height_, width_, CV_32FC1, cv::Scalar(0.f));
  for(int v=0; v<height_; v++)
  {
    for(int u=0; u<width_; u++)
    {
      Eigen::Vector2d uv_dist;
      uv_dist << u, v;
      Eigen::Vector3d bearing = PixelToBearing(uv_dist);
      Eigen::Vector2d uv_dist2 = BearingToPixel(bearing);
      Eigen::Vector2d uv_undist;
      uv_undist[0] = projection_params_[0]*bearing[0]/bearing[2] + projection_params_[2];
      uv_undist[1] = projection_params_[1]*bearing[1]/bearing[2] + projection_params_[3];
      Eigen::Vector2d delta2 = uv_dist2 - uv_dist;
      Eigen::Vector2d delta_un = uv_undist - uv_dist;
      undist_error_.at<cv::Vec2d>(v, u) = cv::Vec2d(delta2.x(), delta2.y());
      dist_delta_.at<cv::Vec2d>(v, u) = cv::Vec2d(delta_un.x(), delta_un.y());
    }
  }
  for(int v=0; v<height_; v++)
  {
    for(int u=0; u<width_; u++)
    {
      cv::Vec2d delta_max(0,0);
      for(int dv=-1; dv<=1; dv++)
      {
        for(int du=-1; du<=1; du++)
        {
          int v1 = v+dv;
          int u1 = u+du;
          if(v1>=0 && v1<height_ && u1>=0 && u1<width_)
          {
            cv::Vec2d delta = dist_delta_.at<cv::Vec2d>(v1, u1) - dist_delta_.at<cv::Vec2d>(v, u);
            if(cv::norm(delta)>cv::norm(delta_max))
            {
              delta_max = delta;
            }
          }
        }
      }
      dist_delta_unconsistency_.at<cv::Vec2d>(v, u) = delta_max;
    }
  }
  for(int v=0; v<height_; v++)
  {
    for(int u=0; u<width_; u++)
    {
      double std1 = cv::norm(undist_error_.at<cv::Vec2d>(v, u));
      double std2 = cv::norm(dist_delta_unconsistency_.at<cv::Vec2d>(v, u));
      double std = std1 + std2 + 0.5;
      pixel_std_.at<float>(v,u) = std;
      if((max_error<0 || std1 < max_error) && (max_unconsistency<0 || std2 < max_unconsistency))
      {
        pixel_weight_.at<float>(v,u) = 1./std;
      }
      else
      {
        pixel_weight_.at<float>(v,u) = 0.f;
      }
    }
  }
  weight_available_ = true;
}

} //  namespace slam_cam