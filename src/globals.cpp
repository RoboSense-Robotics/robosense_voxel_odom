#include "globals.h"

Eigen::Vector4d min_point;
double min_eigen_value;
int max_layer = 2;
int max_points = 100;
double voxel_size = 1.0;
int min_ba_point = 20;
std::vector<double> plane_eigen_value_thre;
double imu_coef = 1e-4;
int* mp;

double imupre_scale_gravity = 1.0;
Eigen::Matrix<double, 6, 6> noiseMeas, noiseWalk;

Eigen::Matrix3d I33(Eigen::Matrix3d::Identity());