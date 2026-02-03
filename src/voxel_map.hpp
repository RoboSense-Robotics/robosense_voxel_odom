#ifndef VOXEL_MAP2_HPP
#define VOXEL_MAP2_HPP

#include "globals.h"
#include "tools.hpp"
#include "preintegration.hpp"
#include "camera_model.h"
#include <thread>
#include <Eigen/Eigenvalues>
#include <unordered_set>
#include <mutex>

#include <fstream>

struct pointVar 
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3d pnt;
  Eigen::Matrix3d var;
  float intensity = 0;
};

using PVec = std::vector<pointVar>;
using PVecPtr = shared_ptr<vector<pointVar>>;

void down_sampling_pvec(PVec &pvec, double voxel_size, pcl::PointCloud<PointType> &pl_keep);


struct Plane
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  Eigen::Vector3d normal = Eigen::Vector3d::Zero();
  Eigen::Vector3d eig_value = Eigen::Vector3d::Zero();
  Eigen::Matrix3d eig_vector = Eigen::Matrix3d::Zero();
  Eigen::Matrix<double, 6, 6> plane_var;
  float radius = 0;
  bool is_plane = false;

  Plane()
  {
    plane_var.setZero();
  }

};


void Bf_var(const pointVar &pv, Eigen::Matrix<double, 9, 9> &bcov, const Eigen::Vector3d &vec);


struct VisualFactorParam
{
  Eigen::Matrix3d rot_cam_to_imu;
  Eigen::Vector3d pos_cam_in_imu;
  double weight_scale_unit;
  bool use_kernel = false;
  double (*loss_function)(double res, double delta) = nullptr;
  double kernel_delta = 1.;
  double reporjection_coeff = -1.;
  std::shared_ptr<const slam_cam::CameraBase> camera_ptr;
  int border_x = 0;
  int border_y = 0;
  double border_weight = 0.;
  bool use_dist_weight = false;
  
  void LogInfo()
  {
    std::cout << "rot_cam_to_imu:\n" << rot_cam_to_imu << std::endl;
    std::cout << "pos_cam_in_imu:" << pos_cam_in_imu.transpose() << std::endl;
    std::cout << "weight_scale_unit:" << weight_scale_unit << std::endl;
    std::cout << "use_kernel:" << use_kernel << ", loss_function:" << loss_function << ", kernel_delta:" << kernel_delta << std::endl;
    std::cout << "reporjection_coeff:" << reporjection_coeff << std::endl;
    std::cout << "camera_ptr:" << camera_ptr << std::endl;
    std::cout << "border_x:" << border_x << ", border_y:" << border_y << ", border_weight:" << border_weight << std::endl;
    std::cout << "use_dist_weight:" << use_dist_weight << std::endl;
  }
};

class VisualFactor
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  VisualFactor(int win_size): win_size_(win_size){}
  VisualFactor(int win_size, VisualFactorParam params)
  : win_size_(win_size), params_(params)
  {
    params_.LogInfo();
  }
  void SetParams(VisualFactorParam params)
  {
    params_ = params;
    params_.LogInfo();
  }
  int size() const
  {
    return feat_pws_.size();
  }
  void clear()
  {
    feat_pws_.clear();
    feat_frame_bearings_.clear();
    feat_frame_uvs_.clear();
    feat_coeffs_.clear();
    feat_ids_.clear();
  }
  ~VisualFactor() {}

  int win_size_;
  VisualFactorParam params_;
  bool lidar_degrade_{false};
  double k_no_degrade_{1.};
  double position_disable_thre__{300.};

  std::vector<Eigen::Vector3d> feat_pws_;
  std::vector<std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>> feat_frame_bearings_;
  std::vector<std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>> feat_frame_uvs_;
  std::vector<double> feat_coeffs_;
  std::vector<int> feat_ids_;

  void push_feat(Eigen::Vector3d pw, 
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> &bearings, 
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> &uvs,
    double coe, int id)
  {
    feat_pws_.push_back(pw);
    feat_frame_bearings_.push_back(bearings);
    feat_frame_uvs_.push_back(uvs);
    feat_coeffs_.push_back(coe);
    feat_ids_.push_back(id);
  }

  void acc_evaluate2(const std::vector<IMUST> &xs, int head, int end, Eigen::MatrixXd &jtj, Eigen::VectorXd &jtb, double &residual)
  {
    residual = 0;
    jtj = Eigen::MatrixXd::Zero(win_size_*6, win_size_*6);
    jtb = Eigen::VectorXd::Zero(win_size_*6);
    double k_degrade = k_no_degrade_;
    if(lidar_degrade_)
    {
      k_degrade = 1.;
    }
    for(int a=head; a<end; a++)
    {
      double feat_res = 0.;
      Eigen::MatrixXd feat_jtj = Eigen::MatrixXd::Zero(win_size_*6, win_size_*6);
      Eigen::VectorXd feat_jtb = Eigen::VectorXd::Zero(win_size_*6);
      int obs_num{0};

      double feat_coe = params_.reporjection_coeff > 0. ? params_.reporjection_coeff : feat_coeffs_[a];
      assert(feat_coe>0.);
      Eigen::Vector3d pw = feat_pws_[a];
      const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> &bearings = feat_frame_bearings_[a];
      const std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> &uvs = feat_frame_uvs_[a];
      
      for(int i=0; i<win_size_; i++)
      {
        if(bearings[i]!=Eigen::Vector3d::Zero())
        {
          Eigen::Matrix3d rot_cam_to_world = xs[i].R * params_.rot_cam_to_imu;
          Eigen::Vector3d pos_cam_in_world = xs[i].R * params_.pos_cam_in_imu + xs[i].p;
          Eigen::Matrix3d rot_world_to_cam = rot_cam_to_world.transpose();
          Eigen::Vector3d pos_world_in_cam = rot_cam_to_world.transpose() * (-pos_cam_in_world);
          Eigen::Vector3d pc = rot_world_to_cam * pw + pos_world_in_cam;

          Eigen::Vector2d obs = bearings[i].head(2)/bearings[i][2];
          Eigen::Vector2d res = params_.weight_scale_unit * (pc.head(2) / pc(2) - obs);

          double weight = (params_.use_kernel && (params_.loss_function!=nullptr)) ? (params_.loss_function(res.norm(),params_.kernel_delta)) : 1.;
          if(!params_.camera_ptr->IsInBorder(uvs[i], params_.border_x, params_.border_y))
          {
            if(params_.border_weight>0.)
            {
              weight *= params_.border_weight;
            }
            else
            {
              continue;
            }
          }
          if(params_.use_dist_weight)
          {
            weight *= params_.camera_ptr->GetPixelWeight(uvs[i]);
          }
          weight *= k_degrade;
          res *= weight;
          Eigen::Matrix<double, 2, 3> J_dres_dpc;
          J_dres_dpc <<
            1./pc(2), 0., -pc(0)/(pc(2)*pc(2)),
            0., 1./pc(2), -pc(1)/(pc(2)*pc(2));
          Eigen::Matrix<double, 2, 6> jac;
          jac.block<2,3>(0,0) = weight * params_.weight_scale_unit * J_dres_dpc * params_.rot_cam_to_imu.transpose() * SkewSymmetric(xs[i].R.transpose() * (pw - xs[i].p));
          jac.block<2,3>(0,3) = - weight * params_.weight_scale_unit * J_dres_dpc * rot_cam_to_world.transpose();
          if(pc(2)>position_disable_thre__)
          {
            jac.block<2,3>(0,3).setZero();
          }
          
          feat_jtj.block<6,6>(i*6, i*6) += jac.transpose() * jac;
          feat_jtb.block<6,1>(i*6, 0) += jac.transpose() * res;
          feat_res += res.dot(res);
          obs_num++;
        }
      }
      if(obs_num>0)
      {
        double scale_normalize = 1./(obs_num*obs_num);
        jtj += feat_coe * scale_normalize * feat_jtj;
        jtb += feat_coe * scale_normalize * feat_jtb;
        residual += feat_coe * scale_normalize * feat_res;
      }
    }
  }

  void evaluate_only_residual(const std::vector<IMUST> &xs, int head, int end, double &residual)
  {
    residual = 0;
      
    for(int a=head; a<end; a++)
    {
      double feat_res = 0.;
      int obs_num{0};

      double feat_coe = params_.reporjection_coeff > 0. ? params_.reporjection_coeff : feat_coeffs_[a];
      assert(feat_coe>0.);
      Eigen::Vector3d pw = feat_pws_[a];
      const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> &bearings = feat_frame_bearings_[a];
      const std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> &uvs = feat_frame_uvs_[a];
      
      for(int i=0; i<win_size_; i++)
      {
        if(bearings[i]!=Eigen::Vector3d::Zero())
        {
          Eigen::Matrix3d rot_cam_to_world = xs[i].R * params_.rot_cam_to_imu;
          Eigen::Vector3d pos_cam_in_world = xs[i].R * params_.pos_cam_in_imu + xs[i].p;
          Eigen::Matrix3d rot_world_to_cam = rot_cam_to_world.transpose();
          Eigen::Vector3d pos_world_in_cam = rot_cam_to_world.transpose() * (-pos_cam_in_world);
          Eigen::Vector3d pc = rot_world_to_cam * pw + pos_world_in_cam;

          Eigen::Vector2d obs = bearings[i].head(2)/bearings[i][2];
          Eigen::Vector2d res = params_.weight_scale_unit * (pc.head(2) / pc(2) - obs);

          double weight = (params_.use_kernel && (params_.loss_function!=nullptr)) ? (params_.loss_function(res.norm(),params_.kernel_delta)) : 1.;
          if(!params_.camera_ptr->IsInBorder(uvs[i], params_.border_x, params_.border_y))
          {
            if(params_.border_weight>0.)
            {
              weight *= params_.border_weight;
            }
            else
            {
              continue;
            }
          }
          if(params_.use_dist_weight)
          {
            weight *= params_.camera_ptr->GetPixelWeight(uvs[i]);
          }
          res *= weight;
          feat_res += res.dot(res);
          obs_num++;
        }
      }
      if(obs_num>0)
      {
        double scale_normalize = 1./(obs_num*obs_num);
        residual += feat_coe * scale_normalize * feat_res;
      }
    }
  }

private:
};

// The LiDAR BA factor in optimization
class LidarFactor
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  std::vector<PointCluster> sig_vecs;
  std::vector<vector<PointCluster>> plvec_voxels;
  std::vector<double> coeffs;
  PLV(3) eig_values; PLM(3) eig_vectors; 
  std::vector<PointCluster> pcr_adds;
  int win_size;

  LidarFactor(int _w): win_size(_w){}

  void push_voxel(vector<PointCluster> &vec_orig, PointCluster &fix, double coe, Eigen::Vector3d &eig_value, Eigen::Matrix3d &eig_vector, PointCluster &pcr_add)
  {
    plvec_voxels.push_back(vec_orig);
    sig_vecs.push_back(fix);
    coeffs.push_back(coe);
    eig_values.push_back(eig_value);
    eig_vectors.push_back(eig_vector);
    pcr_adds.push_back(pcr_add);
  }

  void acc_evaluate2(const std::vector<IMUST> &xs, int head, int end, Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT, double &residual)
  {
    Hess.setZero(); JacT.setZero(); residual = 0;
    std::vector<PointCluster> sig_tran(win_size);
    const int kk = 0;

    PLV(3) viRiTuk(win_size);
    PLM(3) viRiTukukT(win_size);

    std::vector<Eigen::Matrix<double, 3, 6>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 6>>> Auk(win_size);
    Eigen::Matrix3d umumT;

    for(int a=head; a<end; a++)
    {
      std::vector<PointCluster> &sig_orig = plvec_voxels[a];
      double coe = coeffs[a];

      Eigen::Vector3d lmbd = eig_values[a];
      Eigen::Matrix3d U = eig_vectors[a];
      int NN = pcr_adds[a].N;
      Eigen::Vector3d vBar = pcr_adds[a].v / NN;
      
      Eigen::Vector3d u[3] = {U.col(0), U.col(1), U.col(2)};
      Eigen::Vector3d &uk = u[kk];
      Eigen::Matrix3d ukukT = uk * uk.transpose();
      umumT.setZero();
      for(int i=0; i<3; i++)
        if(i != kk)
          umumT += 2.0/(lmbd[kk] - lmbd[i]) * u[i] * u[i].transpose();

      for(int i=0; i<win_size; i++)
      if(sig_orig[i].N != 0)
      {
        Eigen::Matrix3d Pi = sig_orig[i].P;
        Eigen::Vector3d vi = sig_orig[i].v;
        Eigen::Matrix3d Ri = xs[i].R;
        double ni = sig_orig[i].N;

        Eigen::Matrix3d vihat; vihat << SKEW_SYM_MATRX(vi);
        Eigen::Vector3d RiTuk = Ri.transpose() * uk;
        Eigen::Matrix3d RiTukhat; RiTukhat << SKEW_SYM_MATRX(RiTuk);

        Eigen::Vector3d PiRiTuk = Pi * RiTuk;
        viRiTuk[i] = vihat * RiTuk;
        viRiTukukT[i] = viRiTuk[i] * uk.transpose();
        
        Eigen::Vector3d ti_v = xs[i].p - vBar;
        double ukTti_v = uk.dot(ti_v);

        Eigen::Matrix3d combo1 = hat(PiRiTuk) + vihat * ukTti_v;
        Eigen::Vector3d combo2 = Ri*vi + ni*ti_v;
        Auk[i].block<3, 3>(0, 0) = (Ri*Pi + ti_v*vi.transpose()) * RiTukhat - Ri*combo1;
        Auk[i].block<3, 3>(0, 3) = combo2 * uk.transpose() + combo2.dot(uk) * I33;
        Auk[i] /= NN;

        const Eigen::Matrix<double, 6, 1> &jjt = Auk[i].transpose() * uk;
        JacT.block<6, 1>(6*i, 0) += coe * jjt;

        const Eigen::Matrix3d &HRt = 2.0/NN * (1.0-ni/NN) * viRiTukukT[i];
        Eigen::Matrix<double, 6, 6> Hb = Auk[i].transpose() * umumT * Auk[i];
        Hb.block<3, 3>(0, 0) += 2.0/NN * (combo1 - RiTukhat*Pi) * RiTukhat - 2.0/NN/NN * viRiTuk[i] * viRiTuk[i].transpose() - 0.5*hat(jjt.block<3, 1>(0, 0));
        Hb.block<3, 3>(0, 3) += HRt;
        Hb.block<3, 3>(3, 0) += HRt.transpose();
        Hb.block<3, 3>(3, 3) += 2.0/NN * (ni - ni*ni/NN) * ukukT;

        Hess.block<6, 6>(6*i, 6*i) += coe * Hb;
      }

      for(int i=0; i<win_size-1; i++)
      {
        if(sig_orig[i].N != 0)
        {
          double ni = sig_orig[i].N;
          for(int j=i+1; j<win_size; j++)
          {
            if(sig_orig[j].N != 0)
            {
              double nj = sig_orig[j].N;
              Eigen::Matrix<double, 6, 6> Hb = Auk[i].transpose() * umumT * Auk[j];
              Hb.block<3, 3>(0, 0) += -2.0/NN/NN * viRiTuk[i] * viRiTuk[j].transpose();
              Hb.block<3, 3>(0, 3) += -2.0*nj/NN/NN * viRiTukukT[i];
              Hb.block<3, 3>(3, 0) += -2.0*ni/NN/NN * viRiTukukT[j].transpose();
              Hb.block<3, 3>(3, 3) += -2.0*ni*nj/NN/NN * ukukT;

              Hess.block<6, 6>(6*i, 6*j) += coe * Hb;
            }
          }
        }
      }
      
      residual += coe * lmbd[kk];
    }

    for(int i=1; i<win_size; i++)
      for(int j=0; j<i; j++)
        Hess.block<6, 6>(6*i, 6*j) = Hess.block<6, 6>(6*j, 6*i).transpose();
    
  }

  void evaluate_only_residual(const std::vector<IMUST> &xs, int head, int end, double &residual)
  {
    residual = 0;
    int kk = 0; // The kk-th lambda value

    PointCluster pcr;

    for(int a=head; a<end; a++)
    {
      const std::vector<PointCluster> &sig_orig = plvec_voxels[a];
      PointCluster sig = sig_vecs[a];

      for(int i=0; i<win_size; i++)
      if(sig_orig[i].N != 0)
      {
        pcr.transform(sig_orig[i], xs[i]);
        sig += pcr;
      }

      Eigen::Vector3d vBar = sig.v / sig.N;
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(sig.P/sig.N - vBar * vBar.transpose());
      Eigen::Vector3d lmbd = saes.eigenvalues();

      eig_values[a] = saes.eigenvalues();
      eig_vectors[a] = saes.eigenvectors();
      pcr_adds[a] = sig;

      residual += coeffs[a] * lmbd[kk];
    }
    
  }

  int size() const
  {
    return sig_vecs.size();
  }

  void clear()
  {
    sig_vecs.clear(); plvec_voxels.clear();
    eig_values.clear(); eig_vectors.clear();
    pcr_adds.clear(); coeffs.clear();
  }

  ~LidarFactor(){}

};

// The LM optimizer for LiDAR BA
class Lidar_BA_Optimizer
{
public:
  int win_size, jac_leng, thd_num = 2;

  double divide_thread(vector<IMUST> &x_stats, LidarFactor &voxhess, Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT)
  {
    double residual = 0;
    Hess.setZero(); JacT.setZero();
    PLM(-1) hessians(thd_num); 
    PLV(-1) jacobins(thd_num);

    for(int i=0; i<thd_num; i++)
    {
      hessians[i].resize(jac_leng, jac_leng);
      jacobins[i].resize(jac_leng);
    }

    int tthd_num = thd_num;
    std::vector<double> resis(tthd_num, 0);
    int g_size = voxhess.plvec_voxels.size();
    if(g_size < tthd_num) tthd_num = 1;

    std::vector<thread*> mthreads(tthd_num);
    double part = 1.0 * g_size / tthd_num;
    for(int i=1; i<tthd_num; i++)
      mthreads[i] = new thread(&LidarFactor::acc_evaluate2, &voxhess, x_stats, part*i, part*(i+1), ref(hessians[i]), ref(jacobins[i]), ref(resis[i]));

    for(int i=0; i<tthd_num; i++)
    {
      if(i != 0) mthreads[i]->join();
      else
        voxhess.acc_evaluate2(x_stats, 0, part, hessians[0], jacobins[0], resis[0]);
      Hess += hessians[i];
      JacT += jacobins[i];
      residual += resis[i];
      delete mthreads[i];
    }

    return residual;
  }

  double only_residual(vector<IMUST> &x_stats, LidarFactor &voxhess)
  {
    double residual1 = 0;

    std::vector<double> residuals(thd_num, 0);
    int g_size = voxhess.plvec_voxels.size();
    if(g_size < thd_num)
    {
      printf("Too Less Voxel"); exit(0);
    }
    std::vector<thread*> mthreads(thd_num, nullptr);
    double part = 1.0 * g_size / thd_num;
    for(int i=1; i<thd_num; i++)
      mthreads[i] = new thread(&LidarFactor::evaluate_only_residual, &voxhess, x_stats, part*i, part*(i+1), ref(residuals[i]));

    for(int i=0; i<thd_num; i++)
    {
      if(i != 0) 
        mthreads[i]->join();
      else
        voxhess.evaluate_only_residual(x_stats, part*i, part*(i+1), residuals[i]);
      residual1 += residuals[i];
      delete mthreads[i];
    }

    return residual1;
  }

  void reset_voxhess(vector<IMUST> &x_stats, LidarFactor &voxhess)
  {
    int thd_num = 5;
    std::vector<double> residuals(thd_num, 0);
    int g_size = voxhess.plvec_voxels.size();
    if(g_size < thd_num)
    {
      thd_num = 1;
    }
    std::vector<thread*> mthreads(thd_num, nullptr);
    double part = 1.0 * g_size / thd_num;
    for(int i=1; i<thd_num; i++)
      mthreads[i] = new thread(&LidarFactor::evaluate_only_residual, &voxhess, x_stats, part*i, part*(i+1), ref(residuals[i]));

    for(int i=0; i<thd_num; i++)
    {
      if(i != 0) 
      {
        mthreads[i]->join(); delete mthreads[i];
      }
      else
        voxhess.evaluate_only_residual(x_stats, part*i, part*(i+1), residuals[i]);
    }
  }

  bool damping_iter(vector<IMUST> &x_stats, LidarFactor &voxhess, Eigen::MatrixXd* hess, std::vector<double> &resis, int max_iter = 3, bool is_display = false)
  {
    win_size = voxhess.win_size;
    jac_leng = win_size * 6;

    double u = 0.01, v = 2;
    Eigen::MatrixXd D(jac_leng, jac_leng), Hess(jac_leng, jac_leng);
    Eigen::VectorXd JacT(jac_leng), dxi(jac_leng);
    hess->resize(jac_leng, jac_leng);

    D.setIdentity();
    double residual1, residual2, q;
    bool is_calc_hess = true;
    std::vector<IMUST> x_stats_temp = x_stats;

    bool is_converge = true;

    for(int i=0; i<max_iter; i++)
    {
      if(is_calc_hess)
      {
        residual1 = divide_thread(x_stats, voxhess, Hess, JacT);
        *hess = Hess;
      }

      if(i == 0)
        resis.push_back(residual1);

      Hess.topRows(6).setZero();
      Hess.leftCols(6).setZero();
      Hess.block<6, 6>(0, 0).setIdentity();
      JacT.head(6).setZero();
      
      D.diagonal() = Hess.diagonal();
      dxi = (Hess + u*D).ldlt().solve(-JacT);

      for(int j=0; j<win_size; j++)
      {
        x_stats_temp[j].R = x_stats[j].R * Exp(dxi.block<3, 1>(6*j, 0));
        x_stats_temp[j].p = x_stats[j].p + dxi.block<3, 1>(6*j+3, 0);
      }
      double q1 = 0.5*dxi.dot(u*D*dxi-JacT);

      residual2 = only_residual(x_stats_temp, voxhess);

      q = (residual1-residual2);
      if(is_display)
        printf("iter%d: (%lf %lf) u: %lf v: %.1lf q: %.2lf %lf %lf\n", i, residual1, residual2, u, v, q/q1, q1, q);

      if(q > 0)
      {
        x_stats = x_stats_temp;
        double one_three = 1.0 / 3;

        q = q / q1;
        v = 2;
        q = 1 - pow(2*q-1, 3);
        u *= (q<one_three ? one_three:q);
        is_calc_hess = true;
        std::cout << "accept dxi." << std::endl;
      }
      else
      {
        u = u * v;
        v = 2 * v;
        is_calc_hess = false;
        is_converge = false;
        reset_voxhess(x_stats, voxhess);
        std::cout << "reset_voxhess" << std::endl;
      }

      if(fabs((residual1-residual2)/residual1)<1e-6)
        break;
    }
    resis.push_back(residual2);
    return is_converge;
  }

};

#define DVEL 6
// The LiDAR-Inertial BA optimizer
class LI_BA_Optimizer
{
public:
  int win_size, jac_leng, imu_leng;

  void hess_plus(Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT, Eigen::MatrixXd &hs, Eigen::VectorXd &js)
  {
    for(int i=0; i<win_size; i++)
    {
      JacT.block<DVEL, 1>(i*DIM, 0) += js.block<DVEL, 1>(i*DVEL, 0);
      for(int j=0; j<win_size; j++)
        Hess.block<DVEL, DVEL>(i*DIM, j*DIM) += hs.block<DVEL, DVEL>(i*DVEL, j*DVEL);
    }
  }

  double divide_thread(vector<IMUST> &x_stats, LidarFactor &voxhess, std::deque<IMU_PRE*> &imus_factor, Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT)
  {
    int thd_num = 5;
    double residual = 0;
    Hess.setZero(); JacT.setZero();
    PLM(-1) hessians(thd_num); 
    PLV(-1) jacobins(thd_num);
    std::vector<double> resis(thd_num, 0);

    for(int i=0; i<thd_num; i++)
    {
      hessians[i].resize(jac_leng, jac_leng);
      jacobins[i].resize(jac_leng);
    }

    int tthd_num = thd_num;
    int g_size = voxhess.plvec_voxels.size();
    if(g_size < tthd_num) tthd_num = 1;
    double part = 1.0 * g_size / tthd_num;

    std::vector<thread*> mthreads(tthd_num);
    for(int i=1; i<tthd_num; i++)
      mthreads[i] = new thread(&LidarFactor::acc_evaluate2, &voxhess, x_stats, part*i, part * (i+1), ref(hessians[i]), ref(jacobins[i]), ref(resis[i]));

    Eigen::MatrixXd jtj(2*DIM, 2*DIM);
    Eigen::VectorXd gg(2*DIM);

    for(int i=0; i<win_size-1; i++)
    {
      jtj.setZero(); gg.setZero();
      residual += imus_factor[i]->give_evaluate(x_stats[i], x_stats[i+1], jtj, gg, true);
      Hess.block<DIM*2, DIM*2>(i*DIM, i*DIM) += jtj;
      JacT.block<DIM*2, 1>(i*DIM, 0) += gg;
    }

    Eigen::Matrix<double, DIM, DIM> joc;
    Eigen::Matrix<double, DIM, 1> rr;
    joc.setIdentity(); rr.setZero();

    Hess *= imu_coef;
    JacT *= imu_coef;
    residual *= (imu_coef * 0.5);

    for(int i=0; i<tthd_num; i++)
    {
      if(i != 0) mthreads[i]->join();
      else
        voxhess.acc_evaluate2(x_stats, 0, part, hessians[0], jacobins[0], resis[0]);
      hess_plus(Hess, JacT, hessians[i], jacobins[i]);
      residual += resis[i];
      delete mthreads[i];
    }

    return residual;
  }

  double only_residual(vector<IMUST> &x_stats, LidarFactor &voxhess, std::deque<IMU_PRE*> &imus_factor)
  {
    double residual1 = 0, residual2 = 0;
    Eigen::MatrixXd jtj(2*DIM, 2*DIM);
    Eigen::VectorXd gg(2*DIM);

    int thd_num = 5;
    std::vector<double> residuals(thd_num, 0);
    int g_size = voxhess.plvec_voxels.size();
    if(g_size < thd_num)
    {
      thd_num = 1;
    }
    std::vector<thread*> mthreads(thd_num, nullptr);
    double part = 1.0 * g_size / thd_num;
    for(int i=1; i<thd_num; i++)
      mthreads[i] = new thread(&LidarFactor::evaluate_only_residual, &voxhess, x_stats, part*i, part*(i+1), ref(residuals[i]));

    for(int i=0; i<win_size-1; i++)
      residual1 += imus_factor[i]->give_evaluate(x_stats[i], x_stats[i+1], jtj, gg, false);
    residual1 *= (imu_coef * 0.5);

    for(int i=0; i<thd_num; i++)
    {
      if(i != 0) 
      {
        mthreads[i]->join(); delete mthreads[i];
      }
      else
        voxhess.evaluate_only_residual(x_stats, part*i, part*(i+1), residuals[i]);
      residual2 += residuals[i];
    }

    return (residual1 + residual2);
  }

  void reset_voxhess(vector<IMUST> &x_stats, LidarFactor &voxhess)
  {
    int thd_num = 5;
    std::vector<double> residuals(thd_num, 0);
    int g_size = voxhess.plvec_voxels.size();
    if(g_size < thd_num)
    {
      thd_num = 1;
    }
    std::vector<thread*> mthreads(thd_num, nullptr);
    double part = 1.0 * g_size / thd_num;
    for(int i=1; i<thd_num; i++)
      mthreads[i] = new thread(&LidarFactor::evaluate_only_residual, &voxhess, x_stats, part*i, part*(i+1), ref(residuals[i]));

    for(int i=0; i<thd_num; i++)
    {
      if(i != 0) 
      {
        mthreads[i]->join(); delete mthreads[i];
      }
      else
        voxhess.evaluate_only_residual(x_stats, part*i, part*(i+1), residuals[i]);
    }
  }

  void damping_iter(vector<IMUST> &x_stats, LidarFactor &voxhess, std::deque<IMU_PRE*> &imus_factor, Eigen::MatrixXd* hess, int max_iter = 3)
  {
    win_size = voxhess.win_size;
    jac_leng = win_size * 6;
    imu_leng = win_size * DIM;
    double u = 0.01, v = 2;
    Eigen::MatrixXd D(imu_leng, imu_leng), Hess(imu_leng, imu_leng);
    Eigen::VectorXd JacT(imu_leng), dxi(imu_leng);
    hess->resize(imu_leng, imu_leng);

    D.setIdentity();
    double residual1, residual2, q;
    bool is_calc_hess = true;
    std::vector<IMUST> x_stats_temp = x_stats;

    double hesstime = 0;
    double resitime = 0;
  
    for(int i=0; i<max_iter; i++)
    {
      if(is_calc_hess)
      {
        auto tm = std::chrono::high_resolution_clock::now();
        residual1 = divide_thread(x_stats, voxhess, imus_factor, Hess, JacT);
        hesstime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - tm).count()*1e-6;
        *hess = Hess;
      }
      
      Hess.topRows(DIM).setZero();
      Hess.leftCols(DIM).setZero();
      Hess.block<DIM, DIM>(0, 0).setIdentity();
      JacT.head(DIM).setZero();

      D.diagonal() = Hess.diagonal();
      dxi = (Hess + u*D).ldlt().solve(-JacT);

      for(int j=0; j<win_size; j++)
      {
        x_stats_temp[j].R = x_stats[j].R * Exp(dxi.block<3, 1>(DIM*j, 0));
        x_stats_temp[j].p = x_stats[j].p + dxi.block<3, 1>(DIM*j+3, 0);
        x_stats_temp[j].v = x_stats[j].v + dxi.block<3, 1>(DIM*j+6, 0);
        x_stats_temp[j].bg = x_stats[j].bg + dxi.block<3, 1>(DIM*j+9, 0);
        x_stats_temp[j].ba = x_stats[j].ba + dxi.block<3, 1>(DIM*j+12, 0);
      }

      for(int j=0; j<win_size-1; j++)
        imus_factor[j]->update_state(dxi.block<DIM, 1>(DIM*j, 0));

      double q1 = 0.5 * dxi.dot(u*D*dxi-JacT);

      auto tl1 = std::chrono::high_resolution_clock::now();
      residual2 = only_residual(x_stats_temp, voxhess, imus_factor);
      auto tl2 = std::chrono::high_resolution_clock::now();
      resitime += std::chrono::duration_cast<std::chrono::microseconds>(tl2 - tl1).count()*1e-6;

      q = (residual1-residual2);
      printf("iter%d: (%lf %lf) u: %lf v: %.1lf q: %.2lf %lf %lf\n", i, residual1, residual2, u, v, q/q1, q1, q);

      if(q > 0)
      {
        x_stats = x_stats_temp;
        double one_three = 1.0 / 3;

        q = q / q1;
        v = 2;
        q = 1 - pow(2*q-1, 3);
        u *= (q<one_three ? one_three:q);
        is_calc_hess = true;
        std::cout << "accept dxi." << std::endl;
      }
      else
      {
        u = u * v;
        v = 2 * v;
        is_calc_hess = false;

        for(int j=0; j<win_size-1; j++)
        {
          imus_factor[j]->dbg = imus_factor[j]->dbg_buf;
          imus_factor[j]->dba = imus_factor[j]->dba_buf;
        }
        reset_voxhess(x_stats, voxhess);
        std::cout << "reset_voxhess" << std::endl;

      }

      if(fabs((residual1-residual2)/residual1)<1e-6)
        break;
    }
  }

};

// The LiDAR-Inertial BA optimizer with gravity optimization
class LI_BA_OptimizerGravity
{
public:
  int win_size, jac_leng, imu_leng;

  void hess_plus(Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT, Eigen::MatrixXd &hs, Eigen::VectorXd &js)
  {
    for(int i=0; i<win_size; i++)
    {
      JacT.block<DVEL, 1>(i*DIM, 0) += js.block<DVEL, 1>(i*DVEL, 0);
      for(int j=0; j<win_size; j++)
        Hess.block<DVEL, DVEL>(i*DIM, j*DIM) += hs.block<DVEL, DVEL>(i*DVEL, j*DVEL);
    }
  }

  double divide_thread(vector<IMUST> &x_stats, LidarFactor &voxhess, std::deque<IMU_PRE*> &imus_factor, Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT)
  {
    int thd_num = 5;
    double residual = 0;
    Hess.setZero(); JacT.setZero();
    PLM(-1) hessians(thd_num); 
    PLV(-1) jacobins(thd_num);
    std::vector<double> resis(thd_num, 0);

    for(int i=0; i<thd_num; i++)
    {
      hessians[i].resize(jac_leng, jac_leng);
      jacobins[i].resize(jac_leng);
    }

    int tthd_num = thd_num;
    int g_size = voxhess.plvec_voxels.size();
    if(g_size < tthd_num) tthd_num = 1;
    double part = 1.0 * g_size / tthd_num;

    std::vector<thread*> mthreads(tthd_num);
    for(int i=1; i<tthd_num; i++)
      mthreads[i] = new thread(&LidarFactor::acc_evaluate2, &voxhess, x_stats, part*i, part * (i+1), ref(hessians[i]), ref(jacobins[i]), ref(resis[i]));

    Eigen::MatrixXd jtj(2*DIM+3, 2*DIM+3);
    Eigen::VectorXd gg(2*DIM+3);

    for(int i=0; i<win_size-1; i++)
    {
      jtj.setZero(); gg.setZero();
      residual += imus_factor[i]->give_evaluate_g(x_stats[i], x_stats[i+1], jtj, gg, true);
      Hess.block<DIM*2, DIM*2>(i*DIM, i*DIM) += jtj.block<2*DIM, 2*DIM>(0, 0);
      Hess.block<DIM*2, 3>(i*DIM, imu_leng-3) += jtj.block<2*DIM, 3>(0, 2*DIM);
      Hess.block<3, DIM*2>(imu_leng-3, i*DIM) += jtj.block<3, 2*DIM>(2*DIM,0);
      Hess.block<3, 3>(imu_leng-3, imu_leng-3) += jtj.block<3, 3>(2*DIM, 2*DIM);

      JacT.block<DIM*2, 1>(i*DIM, 0) += gg.head(2*DIM);
      JacT.tail(3) += gg.tail(3);
    }

    Eigen::Matrix<double, DIM, DIM> joc;
    Eigen::Matrix<double, DIM, 1> rr;
    joc.setIdentity(); rr.setZero();

    Hess *= imu_coef;
    JacT *= imu_coef;
    residual *= (imu_coef * 0.5);

    for(int i=0; i<tthd_num; i++)
    {
      if(i != 0) mthreads[i]->join();
      else
        voxhess.acc_evaluate2(x_stats, 0, part, hessians[0], jacobins[0], resis[0]);
      hess_plus(Hess, JacT, hessians[i], jacobins[i]);
      residual += resis[i];
      delete mthreads[i];
    }

    return residual;
  }

  double only_residual(vector<IMUST> &x_stats, LidarFactor &voxhess, std::deque<IMU_PRE*> &imus_factor)
  {
    double residual1 = 0, residual2 = 0;
    Eigen::MatrixXd jtj(2*DIM, 2*DIM);
    Eigen::VectorXd gg(2*DIM);

    int thd_num = 5;
    std::vector<double> residuals(thd_num, 0);
    int g_size = voxhess.plvec_voxels.size();
    if(g_size < thd_num)
    {
      thd_num = 1;
    }
    std::vector<thread*> mthreads(thd_num, nullptr);
    double part = 1.0 * g_size / thd_num;
    for(int i=1; i<thd_num; i++)
      mthreads[i] = new thread(&LidarFactor::evaluate_only_residual, &voxhess, x_stats, part*i, part*(i+1), ref(residuals[i]));

    for(int i=0; i<win_size-1; i++)
      residual1 += imus_factor[i]->give_evaluate_g(x_stats[i], x_stats[i+1], jtj, gg, false);
    residual1 *= (imu_coef * 0.5);

    for(int i=0; i<thd_num; i++)
    {
      if(i != 0) 
      {
        mthreads[i]->join(); delete mthreads[i];
      }
      else
        voxhess.evaluate_only_residual(x_stats, part*i, part*(i+1), residuals[i]);
      residual2 += residuals[i];
    }

    return (residual1 + residual2);
  }

  void reset_voxhess(vector<IMUST> &x_stats, LidarFactor &voxhess)
  {
    int thd_num = 5;
    std::vector<double> residuals(thd_num, 0);
    int g_size = voxhess.plvec_voxels.size();
    if(g_size < thd_num)
    {
      thd_num = 1;
    }
    std::vector<thread*> mthreads(thd_num, nullptr);
    double part = 1.0 * g_size / thd_num;
    for(int i=1; i<thd_num; i++)
      mthreads[i] = new thread(&LidarFactor::evaluate_only_residual, &voxhess, x_stats, part*i, part*(i+1), ref(residuals[i]));

    for(int i=0; i<thd_num; i++)
    {
      if(i != 0) 
      {
        mthreads[i]->join(); delete mthreads[i];
      }
      else
        voxhess.evaluate_only_residual(x_stats, part*i, part*(i+1), residuals[i]);
    }
  }

  void damping_iter(vector<IMUST> &x_stats, LidarFactor &voxhess, std::deque<IMU_PRE*> &imus_factor, std::vector<double> &resis, Eigen::MatrixXd* hess, int max_iter = 2)
  {
    win_size = voxhess.win_size;
    jac_leng = win_size * 6;
    imu_leng = win_size * DIM + 3;
    double u = 0.01, v = 2;
    Eigen::MatrixXd D(imu_leng, imu_leng), Hess(imu_leng, imu_leng);
    Eigen::VectorXd JacT(imu_leng), dxi(imu_leng);

    D.setIdentity();
    double residual1, residual2, q;
    bool is_calc_hess = true;
    std::vector<IMUST> x_stats_temp = x_stats;
    
    for(int i=0; i<max_iter; i++)
    {
      if(is_calc_hess)
      {
        residual1 = divide_thread(x_stats, voxhess, imus_factor, Hess, JacT);
        *hess = Hess;
      }

      if(i == 0)
        resis.push_back(residual1);

      Hess.topRows(6).setZero();
      Hess.leftCols(6).setZero();
      Hess.block<6, 6>(0, 0).setIdentity();
      JacT.head(6).setZero();

      D.diagonal() = Hess.diagonal();
      dxi = (Hess + u*D).ldlt().solve(-JacT);

      x_stats_temp[0].g += dxi.tail(3);
      for(int j=0; j<win_size; j++)
      {
        x_stats_temp[j].R = x_stats[j].R * Exp(dxi.block<3, 1>(DIM*j, 0));
        x_stats_temp[j].p = x_stats[j].p + dxi.block<3, 1>(DIM*j+3, 0);
        x_stats_temp[j].v = x_stats[j].v + dxi.block<3, 1>(DIM*j+6, 0);
        x_stats_temp[j].bg = x_stats[j].bg + dxi.block<3, 1>(DIM*j+9, 0);
        x_stats_temp[j].ba = x_stats[j].ba + dxi.block<3, 1>(DIM*j+12, 0);
        x_stats_temp[j].g = x_stats_temp[0].g;
      }

      for(int j=0; j<win_size-1; j++)
        imus_factor[j]->update_state(dxi.block<DIM, 1>(DIM*j, 0));
      
      double q1 = 0.5 * dxi.dot(u*D*dxi-JacT);
      residual2 = only_residual(x_stats_temp, voxhess, imus_factor);
      q = (residual1-residual2);
      printf("iter%d: (%lf %lf) u: %lf v: %.1lf q: %.2lf %lf %lf\n", i, residual1, residual2, u, v, q/q1, q1, q);
      
      if(q > 0)
      {
        x_stats = x_stats_temp;
        double one_three = 1.0 / 3;

        q = q / q1;
        v = 2;
        q = 1 - pow(2*q-1, 3);
        u *= (q<one_three ? one_three:q);
        is_calc_hess = true;
        std::cout << "accept dxi." << std::endl;
      }
      else
      {
        u = u * v;
        v = 2 * v;
        is_calc_hess = false;

        for(int j=0; j<win_size-1; j++)
        {
          imus_factor[j]->dbg = imus_factor[j]->dbg_buf;
          imus_factor[j]->dba = imus_factor[j]->dba_buf;
        }
        reset_voxhess(x_stats, voxhess);
        std::cout << "reset_voxhess" << std::endl;
      }

      if(fabs((residual1-residual2)/residual1)<1e-6)
        break;

    }
    resis.push_back(residual2);
    
  }

};

// The LiDAR-Visual-Inertial BA optimizer
class LVI_BA_Optimizer
{
public:
  int win_size, jac_leng, imu_leng;

  void hess_plus(Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT, Eigen::MatrixXd &hs, Eigen::VectorXd &js)
  {
    for(int i=0; i<win_size; i++)
    {
      JacT.block<DVEL, 1>(i*DIM, 0) += js.block<DVEL, 1>(i*DVEL, 0);
      for(int j=0; j<win_size; j++)
        Hess.block<DVEL, DVEL>(i*DIM, j*DIM) += hs.block<DVEL, DVEL>(i*DVEL, j*DVEL);
    }
  }

  double divide_thread(vector<IMUST> &x_stats, LidarFactor &voxhess, VisualFactor &visual_factor, std::deque<IMU_PRE*> &imus_factor, Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT)
  {
    // lidar
    int thd_num = 5;
    double residual1 = 0, residual2=0, residual3=0;
    Hess.setZero(); JacT.setZero();
    PLM(-1) hessians(thd_num); 
    PLV(-1) jacobins(thd_num);
    std::vector<double> resis(thd_num, 0);

    for(int i=0; i<thd_num; i++)
    {
      hessians[i].resize(jac_leng, jac_leng);
      jacobins[i].resize(jac_leng);
    }

    int tthd_num = thd_num;
    int g_size = voxhess.plvec_voxels.size();
    if(g_size < tthd_num) tthd_num = 1;
    double part = 1.0 * g_size / tthd_num;

    std::vector<thread*> mthreads(tthd_num);
    for(int i=1; i<tthd_num; i++)
      mthreads[i] = new thread(&LidarFactor::acc_evaluate2, &voxhess, x_stats, part*i, part * (i+1), ref(hessians[i]), ref(jacobins[i]), ref(resis[i]));

    // visual
    int thd_num_vis = 5;
    PLM(-1) hessians_vis(thd_num_vis); 
    PLV(-1) jacobins_vis(thd_num_vis);
    std::vector<double> resis_vis(thd_num_vis, 0);

    for(int i=0; i<thd_num_vis; i++)
    {
      hessians_vis[i].resize(jac_leng, jac_leng);
      jacobins_vis[i].resize(jac_leng);
    }

    int tthd_num_vis = thd_num_vis;
    int g_size_vis = visual_factor.feat_pws_.size();
    if(g_size_vis < tthd_num_vis) tthd_num_vis = 1;
    double part_vis = 1.0 * g_size_vis / tthd_num_vis;

    std::vector<thread*> mthreads_vis(tthd_num_vis);
    for(int i=1; i<tthd_num_vis; i++)
      mthreads_vis[i] = new thread(&VisualFactor::acc_evaluate2, &visual_factor, x_stats, part_vis*i, part_vis * (i+1), ref(hessians_vis[i]), ref(jacobins_vis[i]), ref(resis_vis[i]));

    // imu
    Eigen::MatrixXd jtj(2*DIM, 2*DIM);
    Eigen::VectorXd gg(2*DIM);

    for(int i=0; i<win_size-1; i++)
    {
      jtj.setZero(); gg.setZero();
      residual1 += imus_factor[i]->give_evaluate(x_stats[i], x_stats[i+1], jtj, gg, true);
      Hess.block<DIM*2, DIM*2>(i*DIM, i*DIM) += jtj;
      JacT.block<DIM*2, 1>(i*DIM, 0) += gg;
    }

    Eigen::Matrix<double, DIM, DIM> joc;
    Eigen::Matrix<double, DIM, 1> rr;
    joc.setIdentity(); rr.setZero();

    Hess *= imu_coef;
    JacT *= imu_coef;
    residual1 *= (imu_coef * 0.5);

    // lidar
    for(int i=0; i<tthd_num; i++)
    {
      if(i != 0) mthreads[i]->join();
      else
        voxhess.acc_evaluate2(x_stats, 0, part, hessians[0], jacobins[0], resis[0]);
      hess_plus(Hess, JacT, hessians[i], jacobins[i]);
      residual2 += resis[i];
      delete mthreads[i];
    }

    // visual
    for(int i=0; i<tthd_num_vis; i++)
    {
      if(i != 0) mthreads_vis[i]->join();
      else
        visual_factor.acc_evaluate2(x_stats, 0, part_vis, hessians_vis[0], jacobins_vis[0], resis_vis[0]);
      hess_plus(Hess, JacT, hessians_vis[i], jacobins_vis[i]);
      residual3 += resis_vis[i];
      delete mthreads_vis[i];
    }
    std::cout << "[divide_thread] res_imu:" << residual1 << " res_lidar:" << residual2 << " res_vis:" << residual3 << std::endl;
    return (residual1 + residual2 + residual3);
  }

  double only_residual(vector<IMUST> &x_stats, LidarFactor &voxhess, VisualFactor &visual_factor, std::deque<IMU_PRE*> &imus_factor)
  {
    double residual1 = 0, residual2 = 0, residual3 = 0;
    Eigen::MatrixXd jtj(2*DIM, 2*DIM);
    Eigen::VectorXd gg(2*DIM);

    // lidar
    int thd_num = 5;
    std::vector<double> residuals(thd_num, 0);
    int g_size = voxhess.plvec_voxels.size();
    if(g_size < thd_num)
    {
      thd_num = 1;
    }
    std::vector<thread*> mthreads(thd_num, nullptr);
    double part = 1.0 * g_size / thd_num;
    for(int i=1; i<thd_num; i++)
      mthreads[i] = new thread(&LidarFactor::evaluate_only_residual, &voxhess, x_stats, part*i, part*(i+1), ref(residuals[i]));

    // visual
    int thd_num_vis = 5;
    std::vector<double> residuals_vis(thd_num_vis, 0);
    int g_size_vis = visual_factor.feat_pws_.size();
    if(g_size_vis < thd_num_vis)
    {
      thd_num_vis = 1;
    }
    std::vector<thread*> mthreads_vis(thd_num_vis, nullptr);
    double part_vis = 1.0 * g_size_vis / thd_num_vis;
    for(int i=1; i<thd_num_vis; i++)
      mthreads_vis[i] = new thread(&VisualFactor::evaluate_only_residual, &visual_factor, x_stats, part_vis*i, part_vis*(i+1), ref(residuals_vis[i]));

    // imu
    for(int i=0; i<win_size-1; i++)
      residual1 += imus_factor[i]->give_evaluate(x_stats[i], x_stats[i+1], jtj, gg, false);
    residual1 *= (imu_coef * 0.5);

    // lidar
    for(int i=0; i<thd_num; i++)
    {
      if(i != 0) 
      {
        mthreads[i]->join(); delete mthreads[i];
      }
      else
        voxhess.evaluate_only_residual(x_stats, part*i, part*(i+1), residuals[i]);
      residual2 += residuals[i];
    }

    // visual
    for(int i=0; i<thd_num_vis; i++)
    {
      if(i != 0) 
      {
        mthreads_vis[i]->join(); delete mthreads_vis[i];
      }
      else
        visual_factor.evaluate_only_residual(x_stats, part_vis*i, part_vis*(i+1), residuals_vis[i]);
      residual3 += residuals_vis[i];
    }
    std::cout << "[only_residual] res_imu:" << residual1 << " res_lidar:" << residual2 << " res_vis:" << residual3 << std::endl;
    return (residual1 + residual2 + residual3);
  }

  void reset_voxhess(vector<IMUST> &x_stats, LidarFactor &voxhess)
  {
    int thd_num = 5;
    std::vector<double> residuals(thd_num, 0);
    int g_size = voxhess.plvec_voxels.size();
    if(g_size < thd_num)
    {
      thd_num = 1;
    }
    std::vector<thread*> mthreads(thd_num, nullptr);
    double part = 1.0 * g_size / thd_num;
    for(int i=1; i<thd_num; i++)
      mthreads[i] = new thread(&LidarFactor::evaluate_only_residual, &voxhess, x_stats, part*i, part*(i+1), ref(residuals[i]));

    for(int i=0; i<thd_num; i++)
    {
      if(i != 0) 
      {
        mthreads[i]->join(); delete mthreads[i];
      }
      else
        voxhess.evaluate_only_residual(x_stats, part*i, part*(i+1), residuals[i]);
    }
  }

  void damping_iter(vector<IMUST> &x_stats, LidarFactor &voxhess, VisualFactor &visual_factor, std::deque<IMU_PRE*> &imus_factor, Eigen::MatrixXd* hess, int max_iter = 3)
  {
    win_size = voxhess.win_size;
    jac_leng = win_size * 6;
    imu_leng = win_size * DIM;
    double u = 0.01, v = 2;
    Eigen::MatrixXd D(imu_leng, imu_leng), Hess(imu_leng, imu_leng);
    Eigen::VectorXd JacT(imu_leng), dxi(imu_leng);
    hess->resize(imu_leng, imu_leng);

    D.setIdentity();
    double residual1, residual2, q;
    bool is_calc_hess = true;
    std::vector<IMUST> x_stats_temp = x_stats;

    double hesstime = 0;
    double resitime = 0;
  
    for(int i=0; i<max_iter; i++)
    {
      if(is_calc_hess)
      {
        auto tm = std::chrono::high_resolution_clock::now();
        residual1 = divide_thread(x_stats, voxhess, visual_factor, imus_factor, Hess, JacT);
        hesstime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - tm).count()*1e-6;
        *hess = Hess;
      }
      
      Hess.topRows(DIM).setZero();
      Hess.leftCols(DIM).setZero();
      Hess.block<DIM, DIM>(0, 0).setIdentity();
      JacT.head(DIM).setZero();

      D.diagonal() = Hess.diagonal();
      dxi = (Hess + u*D).ldlt().solve(-JacT);

      for(int j=0; j<win_size; j++)
      {
        x_stats_temp[j].R = x_stats[j].R * Exp(dxi.block<3, 1>(DIM*j, 0));
        x_stats_temp[j].p = x_stats[j].p + dxi.block<3, 1>(DIM*j+3, 0);
        x_stats_temp[j].v = x_stats[j].v + dxi.block<3, 1>(DIM*j+6, 0);
        x_stats_temp[j].bg = x_stats[j].bg + dxi.block<3, 1>(DIM*j+9, 0);
        x_stats_temp[j].ba = x_stats[j].ba + dxi.block<3, 1>(DIM*j+12, 0);
      }

      for(int j=0; j<win_size-1; j++)
        imus_factor[j]->update_state(dxi.block<DIM, 1>(DIM*j, 0));

      double q1 = 0.5 * dxi.dot(u*D*dxi-JacT);

      auto tl1 = std::chrono::high_resolution_clock::now();
      residual2 = only_residual(x_stats_temp, voxhess, visual_factor, imus_factor);
      auto tl2 = std::chrono::high_resolution_clock::now();
      resitime += std::chrono::duration_cast<std::chrono::microseconds>(tl2 - tl1).count()*1e-6;

      q = (residual1-residual2);
      printf("iter%d: (%lf %lf) u: %lf v: %.1lf q: %.2lf %lf %lf\n", i, residual1, residual2, u, v, q/q1, q1, q);

      if(q > 0)
      {
        x_stats = x_stats_temp;
        double one_three = 1.0 / 3;

        q = q / q1;
        v = 2;
        q = 1 - pow(2*q-1, 3);
        u *= (q<one_three ? one_three:q);
        is_calc_hess = true;
        std::cout << "accept dxi." << std::endl;
      }
      else
      {
        u = u * v;
        v = 2 * v;
        is_calc_hess = false;

        for(int j=0; j<win_size-1; j++)
        {
          imus_factor[j]->dbg = imus_factor[j]->dbg_buf;
          imus_factor[j]->dba = imus_factor[j]->dba_buf;
        }
        reset_voxhess(x_stats, voxhess);
        std::cout << "reset_voxhess" << std::endl;

      }

      if(fabs((residual1-residual2)/residual1)<1e-6)
        break;
    }
    // printf("ba: %lf %lf %zu\n", hesstime, resitime, voxhess.plvec_voxels.size());

  }

};

// 10 scans merge into a keyframe
struct Keyframe
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  IMUST x0;
  pcl::PointCloud<PointType>::Ptr plptr;
  int exist;
  int id, mp;
  float jour;

  Keyframe(IMUST &_x0): x0(_x0), exist(0)
  {
    plptr.reset(new pcl::PointCloud<PointType>());
  }

  void generate(pcl::PointCloud<PointType> &pl_send, Eigen::Matrix3d rot = Eigen::Matrix3d::Identity(), Eigen::Vector3d tra = Eigen::Vector3d(0, 0, 0))
  {
    Eigen::Vector3d v3;
    for(PointType ap: plptr->points)
    {
      v3 << ap.x, ap.y, ap.z;
      v3 = rot * v3 + tra;
      ap.x = v3[0]; ap.y = v3[1]; ap.z = v3[2];
      pl_send.push_back(ap);
    }
  }

};

// The sldingwindow in each voxel nodes
class SlideWindow
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  std::vector<PVec> points;
  std::vector<PointCluster> pcrs_local;

  SlideWindow(int wdsize)
  {
    pcrs_local.resize(wdsize);
    points.resize(wdsize);
    for(int i=0; i<wdsize; i++)
      points[i].reserve(20);
  }

  void resize(int wdsize)
  {
    if(points.size() != wdsize)
    {
      points.resize(wdsize);
      pcrs_local.resize(wdsize);
    }
  }

  void clear()
  {
    int wdsize = points.size();
    for(int i=0; i<wdsize; i++)
    {
      points[i].clear();
      pcrs_local[i].clear();
    }
  }

  size_t mem_size()
  {
    size_t total_size{0};
    for(auto &ps:points)
    {
      total_size += ps.capacity();
    }
    total_size *= sizeof(pointVar);
    total_size += sizeof(PointCluster) * pcrs_local.capacity();
    return total_size;
  }

};

// The octotree map for odometry and local mapping
// You can re-write it in your own project
class OctoTree
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SlideWindow* sw = nullptr;
  PointCluster pcr_add;
  Eigen::Matrix<double, 9, 9> cov_add;

  PointCluster pcr_fix;
  PVec point_fix;

  int layer, octo_state, wdsize;
  OctoTree* leaves[8];
  double voxel_center[3];
  double jour = 0;
  float quater_length;

  Plane plane;
  bool isexist = false;

  Eigen::Vector3d eig_value;
  Eigen::Matrix3d eig_vector;

  int last_num = 0, opt_state = -1;
  mutex mVox;

  bool latest_lio = false;

  OctoTree(int _l, int _w) : layer(_l), wdsize(_w), octo_state(0)
  {
    for(int i=0; i<8; i++) leaves[i] = nullptr;
    cov_add.setZero();
  }

  void tras_size(size_t &count, size_t &mem_running_size)
  {
    count++;
    if(sw != nullptr)
    {
      mem_running_size += sw->mem_size();
    }
    mem_running_size += sizeof(pointVar) * point_fix.size();
    for(int i=0; i<8; i++)
        if(leaves[i] != nullptr)
          leaves[i]->tras_size(count, mem_running_size);
  }

  inline void push(int ord, const pointVar &pv, const Eigen::Vector3d &pw, std::vector<SlideWindow*> &sws)
  {
    mVox.lock();
    if(sw == nullptr)
    {
      if(sws.size() != 0)
      {
        sw = sws.back();
        sws.pop_back();
        sw->resize(wdsize);
      }
      else
        sw = new SlideWindow(wdsize);
    }
    if(!isexist) isexist = true;

    int mord = mp[ord];
    if(layer < max_layer)
      sw->points[mord].push_back(pv);
    sw->pcrs_local[mord].push(pv.pnt);
    pcr_add.push(pw);
    Eigen::Matrix<double, 9, 9> Bi;
    Bf_var(pv, Bi, pw);
    cov_add += Bi;
    mVox.unlock();
  }

  inline void push_fix(pointVar &pv)
  {
    if(layer < max_layer)
      point_fix.push_back(pv);
    pcr_fix.push(pv.pnt);
    pcr_add.push(pv.pnt);
    Eigen::Matrix<double, 9, 9> Bi;
    Bf_var(pv, Bi, pv.pnt);
    cov_add += Bi;
  }

  inline void push_fix_novar(pointVar &pv)
  {
    if(layer < max_layer)
      point_fix.push_back(pv);
    pcr_fix.push(pv.pnt);
    pcr_add.push(pv.pnt);
  }

  inline bool plane_judge(Eigen::Vector3d &eig_values)
  {
    return (eig_values[0] < min_eigen_value && (eig_values[0]/eig_values[2])<plane_eigen_value_thre[layer]);
  }

  void allocate(int ord, const pointVar &pv, const Eigen::Vector3d &pw, std::vector<SlideWindow*> &sws)
  {
    if(octo_state == 0)
    {
      push(ord, pv, pw, sws);
    }
    else
    {
      int xyz[3] = {0, 0, 0};
      for(int k=0; k<3; k++)
        if(pw[k] > voxel_center[k]) xyz[k] = 1;
      int leafnum = 4*xyz[0] + 2*xyz[1] + xyz[2];

      if(leaves[leafnum] == nullptr)
      {
        leaves[leafnum] = new OctoTree(layer+1, wdsize); 
        leaves[leafnum]->voxel_center[0] = voxel_center[0] + (2*xyz[0]-1)*quater_length;
        leaves[leafnum]->voxel_center[1] = voxel_center[1] + (2*xyz[1]-1)*quater_length;
        leaves[leafnum]->voxel_center[2] = voxel_center[2] + (2*xyz[2]-1)*quater_length;
        leaves[leafnum]->quater_length = quater_length / 2;
      }

      leaves[leafnum]->allocate(ord, pv, pw, sws);
    }

  }

  void allocate_fix(pointVar &pv)
  {
    if(octo_state == 0)
    {
      push_fix_novar(pv);
    }
    else if(layer < max_layer)
    {
      int xyz[3] = {0, 0, 0};
      for(int k=0; k<3; k++)
        if(pv.pnt[k] > voxel_center[k]) xyz[k] = 1;
      int leafnum = 4*xyz[0] + 2*xyz[1] + xyz[2];

      if(leaves[leafnum] == nullptr)
      {
        leaves[leafnum] = new OctoTree(layer+1, wdsize);
        leaves[leafnum]->voxel_center[0] = voxel_center[0] + (2*xyz[0]-1)*quater_length;
        leaves[leafnum]->voxel_center[1] = voxel_center[1] + (2*xyz[1]-1)*quater_length;
        leaves[leafnum]->voxel_center[2] = voxel_center[2] + (2*xyz[2]-1)*quater_length;
        leaves[leafnum]->quater_length = quater_length / 2;
      }

      leaves[leafnum]->allocate_fix(pv);
    }
  }

  void fix_divide(vector<SlideWindow*> &sws)
  {
    for(pointVar &pv: point_fix)
    {
      int xyz[3] = {0, 0, 0};
      for(int k=0; k<3; k++)
        if(pv.pnt[k] > voxel_center[k]) xyz[k] = 1;
      int leafnum = 4*xyz[0] + 2*xyz[1] + xyz[2];
      if(leaves[leafnum] == nullptr)
      {
        leaves[leafnum] = new OctoTree(layer+1, wdsize);
        leaves[leafnum]->voxel_center[0] = voxel_center[0] + (2*xyz[0]-1)*quater_length;
        leaves[leafnum]->voxel_center[1] = voxel_center[1] + (2*xyz[1]-1)*quater_length;
        leaves[leafnum]->voxel_center[2] = voxel_center[2] + (2*xyz[2]-1)*quater_length;
        leaves[leafnum]->quater_length = quater_length / 2;
      }

      leaves[leafnum]->push_fix(pv);
    }

  }

  void subdivide(int si, IMUST &xx, std::vector<SlideWindow*> &sws)
  {
    for(pointVar &pv: sw->points[mp[si]])
    {
      Eigen::Vector3d pw = xx.R * pv.pnt + xx.p;
      int xyz[3] = {0, 0, 0};
      for(int k=0; k<3; k++)
        if(pw[k] > voxel_center[k]) xyz[k] = 1;
      int leafnum = 4*xyz[0] + 2*xyz[1] + xyz[2];
      if(leaves[leafnum] == nullptr)
      {
        leaves[leafnum] = new OctoTree(layer+1, wdsize);
        leaves[leafnum]->voxel_center[0] = voxel_center[0] + (2*xyz[0]-1)*quater_length;
        leaves[leafnum]->voxel_center[1] = voxel_center[1] + (2*xyz[1]-1)*quater_length;
        leaves[leafnum]->voxel_center[2] = voxel_center[2] + (2*xyz[2]-1)*quater_length;
        leaves[leafnum]->quater_length = quater_length / 2;
      }

      leaves[leafnum]->push(si, pv, pw, sws);
    }
  }

  void plane_update()
  {
    plane.center = pcr_add.v / pcr_add.N;
    int l = 0;
    Eigen::Vector3d u[3] = {eig_vector.col(0), eig_vector.col(1), eig_vector.col(2)};
    double nv = 1.0 / pcr_add.N;

    Eigen::Matrix<double, 3, 9> u_c; u_c.setZero();
    for(int k=0; k<3; k++)
    if(k != l)
    {
      Eigen::Matrix3d ukl = u[k] * u[l].transpose();
      Eigen::Matrix<double, 1, 9> fkl;
      fkl.head(6) << ukl(0, 0), ukl(1, 0)+ukl(0, 1), ukl(2, 0)+ukl(0, 2), 
                     ukl(1, 1), ukl(1, 2)+ukl(2, 1),           ukl(2, 2);
      fkl.tail(3) = -(u[k].dot(plane.center) * u[l] + u[l].dot(plane.center) * u[k]);
      
      u_c += nv / (eig_value[l]-eig_value[k]) * u[k] * fkl;
    }

    Eigen::Matrix<double, 3, 9> Jc = u_c * cov_add;
    plane.plane_var.block<3, 3>(0, 0) = Jc * u_c.transpose();
    Eigen::Matrix3d Jc_N = nv * Jc.block<3, 3>(0, 6);
    plane.plane_var.block<3, 3>(0, 3) = Jc_N;
    plane.plane_var.block<3, 3>(3, 0) = Jc_N.transpose();
    plane.plane_var.block<3, 3>(3, 3) = nv * nv * cov_add.block<3, 3>(6, 6);
    plane.normal = u[0];
    plane.radius = eig_value[2];
    plane.eig_value = eig_value;
    plane.eig_vector = eig_vector;
    plane.eig_vector.col(2) = plane.eig_vector.col(0).cross(plane.eig_vector.col(1));
  }

  void recut(int win_count, std::vector<IMUST> &x_buf, std::vector<SlideWindow*> &sws)
  {
    if(octo_state == 0)
    {
      if(layer >= 0)
      {
        opt_state = -1;
        if(pcr_add.N <= min_point[layer])
        {
          plane.is_plane = false; return;
        }
        if(!isexist || sw == nullptr) return;

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(pcr_add.cov());
        eig_value  = saes.eigenvalues();
        eig_vector = saes.eigenvectors();
        plane.is_plane = plane_judge(eig_value);

        if(plane.is_plane)
        {
          return;
        }
        else if(layer >= max_layer)
          return;
      }
      
      if(pcr_fix.N != 0)
      {
        fix_divide(sws);
        PVec().swap(point_fix);
      }

      for(int i=0; i<win_count; i++)
        subdivide(i, x_buf[i], sws);

      sw->clear();
      sws.push_back(sw);
      sw = nullptr;
      octo_state = 1;
    }

    for(int i=0; i<8; i++)
      if(leaves[i] != nullptr)
        leaves[i]->recut(win_count, x_buf, sws);

  }

  void RayTracking(Eigen::Vector3d orig, Eigen::Vector3d dir, double min_range, double max_range, const bool ray_tracking_all_map, Eigen::Vector3d &hit_pt, OctoTree* &hit_octo, double &dist)
  {
    if(octo_state == 0)
    {
      if(layer >= 0 && plane.is_plane && (latest_lio || ray_tracking_all_map))
      {
        double range = plane.normal.dot(plane.center-orig)/plane.normal.dot(dir);
        if(range>min_range && range<max_range)
        {
          Eigen::Matrix3d cov_inv = pcr_add.cov().inverse();
          Eigen::Vector3d delta = orig + dir*range - plane.center;
          double dist_tmp = delta.transpose() * cov_inv * delta;
          if(nullptr == hit_octo || dist_tmp < dist)
          {
            hit_pt = orig + dir*range;
            hit_octo = this;
            dist = dist_tmp;
          }
        }
      }
    }
    else
    {
      for(int i=0; i<8; i++)
      {
        if(leaves[i] != nullptr)
        {
          leaves[i]->RayTracking(orig, dir, min_range, max_range, ray_tracking_all_map, hit_pt, hit_octo, dist);
        }
      }
    }
  }

  void GetPlane(OctoTree *&plane_octo)
  {
    if(plane_octo != nullptr)
    {
      return;
    }
    if(octo_state == 0)
    {
      if(layer >= 0 && plane.is_plane)
      {
        plane_octo = this;
      }
    }
    else
    {
      for(int i=0; i<8; i++)
      {
        if(leaves[i] != nullptr)
        {
          leaves[i]->GetPlane(plane_octo);
        }
      }
    }
  }

  void GetPlane(std::vector<OctoTree*> &plane_octos)
  {
    if(octo_state == 0)
    {
      if(layer >= 0 && plane.is_plane)
      {
        plane_octos.emplace_back(this);
      }
    }
    else
    {
      for(int i=0; i<8; i++)
      {
        if(leaves[i] != nullptr)
        {
          leaves[i]->GetPlane(plane_octos);
        }
      }
    }
  }

  void margi(int win_count, int mgsize, std::vector<IMUST> &x_buf, const LidarFactor &vox_opt)
  {
    if(octo_state == 0 && layer>=0)
    {
      if(!isexist || sw == nullptr) return;
      mVox.lock();
      std::vector<PointCluster> pcrs_world(wdsize);

      if(opt_state >= int(vox_opt.pcr_adds.size()))
      {
        printf("Error: opt_state: %d %zu\n", opt_state, vox_opt.pcr_adds.size());
        exit(0);
      }

      if(opt_state >= 0)
      {
        pcr_add = vox_opt.pcr_adds[opt_state];
        eig_value  = vox_opt.eig_values[opt_state];
        eig_vector = vox_opt.eig_vectors[opt_state];
        opt_state = -1;
        
        for(int i=0; i<mgsize; i++)
        if(sw->pcrs_local[mp[i]].N != 0)
        {
          pcrs_world[i].transform(sw->pcrs_local[mp[i]], x_buf[i]);
        }
      }
      else
      {
        pcr_add = pcr_fix;
        for(int i=0; i<win_count; i++)
        if(sw->pcrs_local[mp[i]].N != 0)
        {
          pcrs_world[i].transform(sw->pcrs_local[mp[i]], x_buf[i]);
          pcr_add += pcrs_world[i];
        }

        if(plane.is_plane)
        {
          Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(pcr_add.cov());
          eig_value = saes.eigenvalues();
          eig_vector = saes.eigenvectors();
        }
        
      }

      if(pcr_fix.N < max_points && plane.is_plane)
      if(pcr_add.N - last_num >= 5 || last_num <= 10)
      {
        plane_update();
        last_num = pcr_add.N;
      }

      if(pcr_fix.N < max_points)
      {
        for(int i=0; i<mgsize; i++)
        if(pcrs_world[i].N != 0)
        {
          pcr_fix += pcrs_world[i];
          for(pointVar pv: sw->points[mp[i]])
          {
            pv.pnt = x_buf[i].R * pv.pnt + x_buf[i].p;
            point_fix.push_back(pv);
          }
        }

      }
      else
      {
        for(int i=0; i<mgsize; i++)
          if(pcrs_world[i].N != 0)
            pcr_add -= pcrs_world[i];
        
        if(point_fix.size() != 0)
          PVec().swap(point_fix);
      }

      for(int i=0; i<mgsize; i++)
      if(sw->pcrs_local[mp[i]].N != 0)
      {
        sw->pcrs_local[mp[i]].clear();
        sw->points[mp[i]].clear();
      }
      
      if(pcr_fix.N >= pcr_add.N)
        isexist = false;
      else
        isexist = true;
      
      mVox.unlock();
    }
    else
    {
      isexist = false;
      for(int i=0; i<8; i++)
      if(leaves[i] != nullptr)
      {
        leaves[i]->margi(win_count, mgsize, x_buf, vox_opt);
        isexist = isexist || leaves[i]->isexist;
      }
    }

  }

  // Extract the LiDAR factor
  void tras_opt(LidarFactor &vox_opt)
  {
    if(octo_state == 0)
    {
      if(layer >= 0 && isexist && plane.is_plane && sw!=nullptr)
      {
        if(eig_value[0]/eig_value[1] > 0.12) return;

        double coe = 1;
        std::vector<PointCluster> pcrs(wdsize);
        for(int i=0; i<wdsize; i++)
          pcrs[i] = sw->pcrs_local[mp[i]];
        opt_state = vox_opt.plvec_voxels.size();
        vox_opt.push_voxel(pcrs, pcr_fix, coe, eig_value, eig_vector, pcr_add);
      }

    }
    else
    {
      for(int i=0; i<8; i++)
        if(leaves[i] != nullptr)
          leaves[i]->tras_opt(vox_opt);
    }


  }

  int match(Eigen::Vector3d &wld, Plane* &pla, double &max_prob, Eigen::Matrix3d &var_wld, double &sigma_d, OctoTree* &oc)
  {
    int flag = 0;
    if(octo_state == 0)
    {
      if(plane.is_plane)
      {
        float dis_to_plane = fabs(plane.normal.dot(wld - plane.center));
        float dis_to_center = (plane.center - wld).squaredNorm();
        float range_dis = (dis_to_center - dis_to_plane * dis_to_plane);
        if(range_dis <= 3*3*plane.radius)
        {
          Eigen::Matrix<double, 1, 6> J_nq;
          J_nq.block<1, 3>(0, 0) = wld - plane.center;
          J_nq.block<1, 3>(0, 3) = -plane.normal;
          double sigma_l = J_nq * plane.plane_var * J_nq.transpose();
          sigma_l += plane.normal.transpose() * var_wld * plane.normal;
          if(dis_to_plane < 3 * sqrt(sigma_l))
          {
            {
              oc = this;
              sigma_d = sigma_l;
              pla = &plane;
            }

            flag = 1;
          }
        }
      }
    }
    else
    {
      int xyz[3] = {0, 0, 0};
      for(int k=0; k<3; k++)
        if(wld[k] > voxel_center[k]) xyz[k] = 1;
      int leafnum = 4*xyz[0] + 2*xyz[1] + xyz[2];

      if(leaves[leafnum] != nullptr)
        flag = leaves[leafnum]->match(wld, pla, max_prob, var_wld, sigma_d, oc);
    }

    return flag;
  }

  void tras_ptr(vector<OctoTree*> &octos_release)
  {
    if(octo_state == 1)
    {
      for(int i=0; i<8; i++)
      if(leaves[i] != nullptr)
      {
        octos_release.push_back(leaves[i]);
        leaves[i]->tras_ptr(octos_release);
      }
    }
  }

  // Extract the point cloud map for debug
  void tras_display(int win_count, pcl::PointCloud<PointType> &pl_fixd, pcl::PointCloud<PointType> &pl_wind, std::vector<IMUST> &x_buf)
  {
    if(octo_state == 0)
    {
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(pcr_add.cov());
      Eigen::Matrix3d eig_vectors = saes.eigenvectors();
      Eigen::Vector3d eig_values  = saes.eigenvalues();

      PointType ap; 

      if(plane.is_plane)
      {
        for(int i=0; i<win_count; i++)
        for(pointVar &pv: sw->points[mp[i]])
        {
          Eigen::Vector3d pvec = x_buf[i].R * pv.pnt + x_buf[i].p;
          ap.x = pvec[0]; ap.y = pvec[1]; ap.z = pvec[2];
          pl_wind.push_back(ap);
        }
      }

    }
    else
    {
      for(int i=0; i<8; i++)
        if(leaves[i] != nullptr)
          leaves[i]->tras_display(win_count, pl_fixd, pl_wind, x_buf);
    }

  }

  bool inside(Eigen::Vector3d &wld)
  {
    double hl = quater_length * 2;
    return (wld[0] >= voxel_center[0] - hl &&
            wld[0] <= voxel_center[0] + hl &&
            wld[1] >= voxel_center[1] - hl &&
            wld[1] <= voxel_center[1] + hl &&
            wld[2] >= voxel_center[2] - hl &&
            wld[2] <= voxel_center[2] + hl);
  }

  void clear_slwd(vector<SlideWindow*> &sws)
  {
    if(octo_state != 0)
    {
      for(int i=0; i<8; i++)
      if(leaves[i] != nullptr)
      {
        leaves[i]->clear_slwd(sws);
      }
    }

    if(sw != nullptr)
    {
      sw->clear();
      sws.push_back(sw);
      sw = nullptr;
    }

  }

};

void cut_voxel(unordered_map<VOXEL_LOC, OctoTree*> &feat_map, PVecPtr pvec, int win_count, unordered_map<VOXEL_LOC, OctoTree*> &feat_tem_map, int wdsize, PLV(3) &pwld, std::vector<SlideWindow*> &sws);


// Cut the current scan into corresponding voxel in multi thread
void cut_voxel_multi(unordered_map<VOXEL_LOC, OctoTree*> &feat_map, PVecPtr pvec, int win_count, unordered_map<VOXEL_LOC, OctoTree*> &feat_tem_map, int wdsize, PLV(3) &pwld, std::vector<vector<SlideWindow*>> &sws);

void cut_voxel(unordered_map<VOXEL_LOC, OctoTree*> &feat_map, PVec &pvec, int wdsize, double jour);


// Match the point with the plane in the voxel map
int match(unordered_map<VOXEL_LOC, OctoTree*> &feat_map, Eigen::Vector3d &wld, Plane* &pla, Eigen::Matrix3d &var_wld, double &sigma_d, OctoTree* &oc);


#endif
