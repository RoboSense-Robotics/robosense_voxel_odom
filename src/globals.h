#include <vector>
#include <Eigen/Dense>

extern Eigen::Vector4d min_point;
extern double min_eigen_value;
extern int max_layer;
extern int max_points;
extern double voxel_size;
extern int min_ba_point;
extern std::vector<double> plane_eigen_value_thre;
extern double imu_coef;
extern int* mp;

extern double imupre_scale_gravity;
extern Eigen::Matrix<double, 6, 6> noiseMeas, noiseWalk;

extern Eigen::Matrix3d I33;