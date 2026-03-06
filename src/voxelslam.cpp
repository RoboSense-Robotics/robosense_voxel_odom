#include "voxelslam.hpp"
#include <atomic>
#include "LOAM/lidar_odometry.h"

using namespace std;

class ResultOutput
{
public:
  static ResultOutput &instance()
  {
    static ResultOutput inst;
    return inst;
  }

  void pub_odom_func(IMUST &xc)
  {
    Eigen::Quaterniond q_this(xc.R);
    Eigen::Vector3d t_this = xc.p;

    static RosNode ros_node = ROS_NODE("local_node");
    static RosTransformBroadcaster br = ROS_TRANSFORM_BROADCASTER(ros_node);
    RosTransformStamped transform_stamped;
    transform_stamped.transform.translation.x = t_this.x();
    transform_stamped.transform.translation.y = t_this.y();
    transform_stamped.transform.translation.z = t_this.z();
    transform_stamped.transform.rotation.w  = q_this.w();
    transform_stamped.transform.rotation.x  = q_this.x();
    transform_stamped.transform.rotation.y  = q_this.y();
    transform_stamped.transform.rotation.z  = q_this.z();

    transform_stamped.header.stamp = ROS_TIME_NOW();
    transform_stamped.header.frame_id = "/camera_init";
    transform_stamped.child_frame_id = "/aft_mapped";

    ROS_SEND_TRANSFORM(br, transform_stamped);
  }

  void pub_localtraj(PLV(3) &pwld, double jour, IMUST &x_curr, int cur_session, pcl::PointCloud<PointType> &pcl_path, PVecPtr &pptr)
  {
    pub_odom_func(x_curr);
    if(ROS_GET_SUB_NUM(pub_scan)!=0)
    {
      pcl::PointCloud<PointType> pcl_send;
      pcl_send.reserve(pwld.size());
      // for(Eigen::Vector3d &pw: pwld)
      for(int i=0; i<pwld.size(); i++)
      {
        Eigen::Vector3d pvec = pwld[i];
        PointType ap;
        ap.x = pvec.x();
        ap.y = pvec.y();
        ap.z = pvec.z();
        ap.intensity = pptr->at(i).intensity;
        pcl_send.push_back(ap);
      }
      pub_pl_func(pcl_send, pub_scan);
    }
    
    if(ROS_GET_SUB_NUM(pub_curr_path)!=0)
    {
      Eigen::Vector3d pcurr = x_curr.p;

      PointType ap;
      ap.x = pcurr[0];
      ap.y = pcurr[1];
      ap.z = pcurr[2];
      ap.curvature = jour;
      ap.intensity = cur_session;
      pcl_path.push_back(ap);
      pub_pl_func(pcl_path, pub_curr_path);
    }
  }

};

class Initialization
{
public:
  static Initialization &instance()
  {
    static Initialization inst;
    return inst;
  }

  void align_gravity(vector<IMUST> &xs)
  {
    Eigen::Vector3d g0 = xs[0].g;
    Eigen::Vector3d n0 = g0 / g0.norm();
    Eigen::Vector3d n1(0, 0, 1);
    if(n0[2] < 0)
      n1[2] = -1;
    
    Eigen::Vector3d rotvec = n0.cross(n1);
    double rnorm = rotvec.norm();
    rotvec = rotvec / rnorm;

    Eigen::AngleAxisd angaxis(asin(rnorm), rotvec);
    Eigen::Matrix3d rot = angaxis.matrix();
    g0 = rot * g0;

    Eigen::Vector3d p0 = xs[0].p;
    for(int i=0; i<xs.size(); i++)
    {
      xs[i].p = rot * (xs[i].p - p0) + p0;
      xs[i].R = rot * xs[i].R;
      xs[i].v = rot * xs[i].v;
      xs[i].g = g0;
    }

  }

  void motion_blur(pcl::PointCloud<PointType> &pl, PVec &pvec, IMUST xc, IMUST xl, std::deque<RosImuPtr> &imus, double pcl_beg_time, IMUST &extrin_para)
  {
    xc.bg = xl.bg; xc.ba = xl.ba;
    Eigen::Vector3d acc_imu, angvel_avr, acc_avr, vel_imu(xc.v), pos_imu(xc.p);
    Eigen::Matrix3d R_imu(xc.R);
    std::vector<IMUST> imu_poses;
    for(auto it_imu=imus.end()-1; it_imu!=imus.begin(); it_imu--)
    {
      RosImu &head = **(it_imu-1);
      RosImu &tail = **(it_imu); 
      
      angvel_avr << 0.5*(head.angular_velocity.x + tail.angular_velocity.x), 
                    0.5*(head.angular_velocity.y + tail.angular_velocity.y), 
                    0.5*(head.angular_velocity.z + tail.angular_velocity.z);
      acc_avr << 0.5*(head.linear_acceleration.x + tail.linear_acceleration.x), 
                 0.5*(head.linear_acceleration.y + tail.linear_acceleration.y), 
                 0.5*(head.linear_acceleration.z + tail.linear_acceleration.z);
      
      angvel_avr -= xc.bg;
      acc_avr = acc_avr * imupre_scale_gravity - xc.ba;
      
      double dt = ROS_TIMESTAMP_TO_SEC(head.header.stamp) - ROS_TIMESTAMP_TO_SEC(tail.header.stamp);
      
      Eigen::Matrix3d acc_avr_skew = hat(acc_avr);
      Eigen::Matrix3d Exp_f = Exp(angvel_avr, dt);

      acc_imu = R_imu * acc_avr + xc.g;
      pos_imu = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt;
      vel_imu = vel_imu + acc_imu * dt;
      R_imu = R_imu * Exp_f;

      double offt = ROS_TIMESTAMP_TO_SEC(head.header.stamp) - pcl_beg_time;
      imu_poses.emplace_back(offt, R_imu, pos_imu, vel_imu, angvel_avr, acc_imu);
    }

    pointVar pv; pv.var.setIdentity();
    if(point_notime)
    {
      for(PointType &ap: pl.points)
      {
        pv.pnt << ap.x, ap.y, ap.z;
        pv.pnt = extrin_para.R * pv.pnt + extrin_para.p;
        pvec.push_back(pv);
      }
      return;
    }
    auto it_pcl = pl.end() - 1;
    for(auto it_kp=imu_poses.begin(); it_kp!=imu_poses.end(); it_kp++)
    {
      IMUST &head = *it_kp;
      R_imu = head.R;
      acc_imu = head.ba;
      vel_imu = head.v;
      pos_imu = head.p;
      angvel_avr = head.bg;

      for(; it_pcl->curvature > head.t; it_pcl--)
      {
        double dt = it_pcl->curvature - head.t;
        Eigen::Matrix3d R_i = R_imu * Exp(angvel_avr, dt);
        Eigen::Vector3d T_ei = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - xc.p;

        Eigen::Vector3d P_i(it_pcl->x, it_pcl->y, it_pcl->z);
        Eigen::Vector3d P_compensate = xc.R.transpose() * (R_i * (extrin_para.R * P_i + extrin_para.p) + T_ei);

        pv.pnt = P_compensate;
        pvec.push_back(pv);
        if(it_pcl == pl.begin()) break;
      }

    }
  }

  int motion_init(vector<pcl::PointCloud<PointType>::Ptr> &pl_origs, std::vector<deque<RosImuPtr>> &vec_imus, std::vector<double> &beg_times, Eigen::MatrixXd *hess, LidarFactor &voxhess, std::vector<IMUST> &x_buf, unordered_map<VOXEL_LOC, OctoTree*> &surf_map, unordered_map<VOXEL_LOC, OctoTree*> &surf_map_slide, std::vector<PVecPtr> &pvec_buf, int win_size, std::vector<vector<SlideWindow*>> &sws, IMUST &x_curr, std::deque<IMU_PRE*> &imu_pre_buf, IMUST &extrin_para, int init_mode)
  {
    PLV(3) pwld;
    double last_g_norm = x_buf[0].g.norm();
    int converge_flag = (1==init_mode) ? 1 : 0;

    double min_eigen_value_orig = min_eigen_value;
    std::vector<double> eigen_value_array_orig = plane_eigen_value_thre;

    min_eigen_value = 0.02;
    for(double &iter: plane_eigen_value_thre)
      iter = 1.0 / 4;

    auto t0 = std::chrono::high_resolution_clock::now();
    double converge_thre = 0.05;
    int converge_times = 0;
    bool is_degrade = true;
    Eigen::Vector3d eigvalue; eigvalue.setZero();
    for(int iterCnt = 0; iterCnt < 10; iterCnt++)
    {
      if(converge_flag == 1)
      {
        min_eigen_value = min_eigen_value_orig;
        plane_eigen_value_thre = eigen_value_array_orig;
      }

      std::vector<OctoTree*> octos;
      for(auto iter=surf_map.begin(); iter!=surf_map.end(); ++iter)
      {
        iter->second->tras_ptr(octos);
        iter->second->clear_slwd(sws[0]);
        delete iter->second;
      }
      for(int i=0; i<octos.size(); i++)
        delete octos[i];
      surf_map.clear(); octos.clear(); surf_map_slide.clear();

      for(int i=0; i<win_size; i++)
      {
        pwld.clear();
        pvec_buf[i]->clear();
        int l = i==0 ? i : i - 1;
        motion_blur(*pl_origs[i], *pvec_buf[i], x_buf[i], x_buf[l], vec_imus[i], beg_times[i], extrin_para);

        if(converge_flag == 1)
        {
          for(pointVar &pv: *pvec_buf[i])
            calcBodyVar(pv.pnt, dept_err, beam_err, pv.var);
          pvec_update(pvec_buf[i], x_buf[i], pwld);
        }
        else
        {
          for(pointVar &pv: *pvec_buf[i])
            pwld.push_back(x_buf[i].R * pv.pnt + x_buf[i].p);
        }

        cut_voxel(surf_map, pvec_buf[i], i, surf_map_slide, win_size, pwld, sws[0]);
        pcl::PointCloud<PointType> cloud_world;
        for(int i=0; i<pwld.size(); i++)
        {
          Eigen::Vector3d &pw = pwld[i];
          PointType p;
          p.x = pw.x();
          p.y = pw.y();
          p.z = pw.z();
          cloud_world.points.push_back(p);
        }
        cloud_world.height = 1;
        cloud_world.width = cloud_world.points.size();
      }

      // LidarFactor voxhess(win_size);
      voxhess.clear(); voxhess.win_size = win_size;
      for(auto iter=surf_map.begin(); iter!=surf_map.end(); ++iter)
      {
        iter->second->recut(win_size, x_buf, sws[0]);
        iter->second->tras_opt(voxhess);
      }

      if(voxhess.plvec_voxels.size() < 10)
        break;
      if(1 == init_mode)
      {
        converge_thre = 0.01;
        is_degrade = false;
        break;
      }
      else
      {
        LI_BA_OptimizerGravity opt_lsv;
        std::vector<double> resis;
        opt_lsv.damping_iter(x_buf, voxhess, imu_pre_buf, resis, hess, 3);
        Eigen::Matrix3d nnt; nnt.setZero();

        printf("%d: %lf %lf %lf: %lf %lf\n", iterCnt, x_buf[0].g[0], x_buf[0].g[1], x_buf[0].g[2], x_buf[0].g.norm(), fabs(resis[0] - resis[1]) / resis[0]);

        for(int i=0; i<win_size-1; i++)
          delete imu_pre_buf[i];
        imu_pre_buf.clear();

        for(int i=1; i<win_size; i++)
        {
          imu_pre_buf.push_back(new IMU_PRE(x_buf[i-1].bg, x_buf[i-1].ba));
          imu_pre_buf.back()->push_imu(vec_imus[i]);
        }

        if(fabs(resis[0] - resis[1]) / resis[0] < converge_thre && iterCnt >= 2)
        {
          for(Eigen::Matrix3d &iter: voxhess.eig_vectors)
          {
            Eigen::Vector3d v3 = iter.col(0);
            nnt += v3 * v3.transpose();
          }
          Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(nnt);
          eigvalue = saes.eigenvalues();
          is_degrade = eigvalue[0] < 15 ? true : false;

          converge_thre = 0.01;
          if(converge_flag == 0)
          {
            align_gravity(x_buf);
            converge_flag = 1;
            continue;
          }
          else
            break;
        }
      }
    }
    x_curr = x_buf[win_size - 1];
    double gnm = x_curr.g.norm();
    if(is_degrade || gnm < 9.6 || gnm > 10.0)
    {
      converge_flag = 0;
    }
    if(converge_flag == 0)
    {
      std::vector<OctoTree*> octos;
      for(auto iter=surf_map.begin(); iter!=surf_map.end(); ++iter)
      {
        iter->second->tras_ptr(octos);
        iter->second->clear_slwd(sws[0]);
        delete iter->second;
      }
      for(int i=0; i<octos.size(); i++)
        delete octos[i];
      surf_map.clear(); octos.clear(); surf_map_slide.clear();
    }

    printf("mn: %lf %lf %lf\n", eigvalue[0], eigvalue[1], eigvalue[2]);
    Eigen::Vector3d angv(vec_imus[0][0]->angular_velocity.x, vec_imus[0][0]->angular_velocity.y, vec_imus[0][0]->angular_velocity.z);
    Eigen::Vector3d acc(vec_imus[0][0]->linear_acceleration.x, vec_imus[0][0]->linear_acceleration.y, vec_imus[0][0]->linear_acceleration.z);
    acc *= 9.8;

    pl_origs.clear(); vec_imus.clear(); beg_times.clear();
    auto t1 = std::chrono::high_resolution_clock::now();
    printf("init time: %lf\n", std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()*1e-6);

    pcl::PointCloud<PointType> pcl_send; PointType pt;
    for(int i=0; i<win_size; i++)
    for(pointVar &pv: *pvec_buf[i])
    {
      Eigen::Vector3d vv = x_buf[i].R * pv.pnt + x_buf[i].p;
      pt.x = vv[0]; pt.y = vv[1]; pt.z = vv[2];
      pcl_send.push_back(pt);
    }
    pub_pl_func(pcl_send, pub_init);

    return converge_flag;
  }

};

class VOXEL_SLAM
{
public:
  pcl::PointCloud<PointType> pcl_path;
  IMUST x_curr, extrin_para;
  IMUEKF odom_ekf;
  unordered_map<VOXEL_LOC, OctoTree*> surf_map, surf_map_slide;
  std::vector<std::shared_ptr<slam_cam::Tracking>> tracking_ptrs_curr;
  double down_size;
  std::vector<OctoTree*> latest_lio_octos;

  int win_size;
  std::vector<IMUST> x_buf;
  std::vector<PVecPtr> pvec_buf;
  std::deque<IMU_PRE*> imu_pre_buf;
  int win_count = 0, win_base = 0;
  std::vector<vector<SlideWindow*>> sws;

  IMUST dx;
  std::vector<OctoTree*> octos_release;
  int reset_flag = 0;
  int g_update = 0;
  int thread_num = 5;
  int degrade_bound = 10;
  bool localBA_enable = true;
  int localBA_update_status{0};
  double localBA_reprojection_coeff{-1.};
  int localBA_border_x{0};
  int localBA_border_y{0};
  double localBA_border_weight{0.};
  std::ofstream of_landmark;
  std::array<double,7> landmark_nums;

  bool is_finish = false;

  int odom_max_iter_ = 4;
  int localBA_max_iter_ = 3;

  std::vector<string> sessionNames;
  string bagname, savepath;

  pcl::PointCloud<PointType> cloud_intensity_save;
  pcl::PointCloud<PointRGBType> cloud_rgb_save;

  bool pub_dense_{false};
  double save_map_duration_{-1};
  double map_start_time_{-1.};
  double map_end_time_{-1.};
  std::string map_save_path_;

  bool record_memory_{false};
  int malloc_trim_duration_{-1};
  double max_jour_{-1.};
  std::ofstream memory_log_;

  int sleep_duration_{1};

  cv::Mat range_img_;
  int range_search_radius_{0};
  std::string radius_save_path_;
  Eigen::Matrix3d rot_lidar_to_cam_ = Eigen::Matrix3d::Identity();
  Eigen::Vector3d pos_lidar_in_cam_ = Eigen::Vector3d::Zero();

  bool lidar_degrade_{false};
  double k_no_degrade_{1.};
  double position_disable_thre__{300.};

  std::ofstream time_cost_drop_;
  std::ofstream time_cost_init_;
  std::ofstream time_cost_lvio_;

  std::atomic<bool> odom_alive_{true};

  bool boundary_point_remove_{false};
  std::shared_ptr<RsLidarOdometry> lidar_odometry_{nullptr};

  VOXEL_SLAM(const std::string cfg_path)
  {
    LoadCalibration(cfg_path);

    const std::string cfg_odom = cfg_path + "/odom.yaml";

    YAML::Node odom_node = YAML::LoadFile(cfg_odom);
    YamlRead(odom_node["General"], "sync_mode", sync_mode);
    YamlRead(odom_node["General"], "img_enable", img_enable);
    YamlRead(odom_node["General"], "img_ds_ratio", img_ds_ratio);

    YamlRead(odom_node["General"], "bagname", bagname);
    savepath = std::string(PROJECT_PATH) + "/" + "Log";
    map_save_path_ = savepath + "/" + bagname + "/";
    radius_save_path_ = savepath + "/" + bagname + "/radius/";

    {
      struct stat st = {0};
      if(stat(savepath.c_str(), &st) == -1) 
      {
        if(mkdir(savepath.c_str(), 0755) == -1) 
        {
          printf("Failed to create directory: %s\n", strerror(errno));
          printf("The logs will be saved in %s.\n", savepath.c_str());
          printf("So please clear or rename the existed folder.\n"); 
          exit(1);
        }
      }
    }
    {
      
      struct stat st = {0};
      if(stat(map_save_path_.c_str(), &st) == -1) 
      {
        if(mkdir(map_save_path_.c_str(), 0755) == -1) 
        {
          printf("Failed to create directory: %s\n", strerror(errno));
          printf("The map will be saved in %s.\n", map_save_path_.c_str());
          printf("So please clear or rename the existed folder.\n"); 
          exit(1);
        }
      }
    }
    {
      struct stat st = {0};
      if(stat(radius_save_path_.c_str(), &st) == -1) 
      {
        if(mkdir(radius_save_path_.c_str(), 0755) == -1) 
        {
          printf("Failed to create directory: %s\n", strerror(errno));
          printf("The radius will be saved in %s.\n", radius_save_path_.c_str());
          printf("So please clear or rename the existed folder.\n"); 
          exit(1);
        }
      }
    }

    YamlRead(odom_node["General"], "lidar_type", feat.lidar_type);
    YamlRead(odom_node["General"], "blind", feat.blind);
    YamlRead(odom_node["General"], "blind_far", feat.blind_far);
    YamlRead(odom_node["General"], "point_filter_num", feat.point_filter_num);

    
    std::string enhance_method;
    int num_feats, fast_threshold, grid_x, grid_y, min_px_dist;
    
    YamlRead(odom_node["General"], "enhance_method", enhance_method);
    YamlRead(odom_node["General"], "num_feats", num_feats);
    YamlRead(odom_node["General"], "fast_threshold", fast_threshold);
    YamlRead(odom_node["General"], "grid_x", grid_x);
    YamlRead(odom_node["General"], "grid_y", grid_y);
    YamlRead(odom_node["General"], "min_px_dist", min_px_dist);
    YamlRead(odom_node["General"], "reprojection_cov", reprojection_cov);
    YamlRead(odom_node["General"], "vis_update_status", vis_update_status);
    YamlRead(odom_node["General"], "re_weight", re_weight);
    YamlRead(odom_node["General"], "border_x", border_x);
    YamlRead(odom_node["General"], "border_y", border_y);
    
    border_x = std::max(0, border_x);
    border_y = std::max(0, border_y);

    std::cout << "Original opencv thread num:" << cv::getNumThreads() << std::endl;
    int opencv_thread_num{-1};
    YamlRead(odom_node["General"], "opencv_thread_num", opencv_thread_num);
    cv::setNumThreads(opencv_thread_num);
    std::cout << "Set opencv thread num:" << cv::getNumThreads() << std::endl;

    YamlRead(odom_node["General"], "plane_density", plane_density);
    assert(plane_density.size()==4);
    std::cout << "plane_density:";
    for(auto &d:plane_density)
    {
      std::cout << d << " ";
      d *= M_PI;
    }
    std::cout << std::endl;

    YamlRead(odom_node["General"], "scale_xy", scale_xy_);
    YamlRead(odom_node["General"], "scale_z", scale_z_);
    YamlRead(odom_node["General"], "landmark_freeze_num", slam_cam::Tracking::landmark_freeze_num_);
    YamlRead(odom_node["General"], "ray_tracking_all_map", ray_tracking_all_map);
    std::vector<double> ray_tracking_range_tmp;
    YamlRead(odom_node["General"], "ray_tracking_range", ray_tracking_range_tmp);
    assert(ray_tracking_range_tmp.size()==2);
    ray_tracking_range[0] = ray_tracking_range_tmp[0];
    ray_tracking_range[1] = ray_tracking_range_tmp[1];
    YamlRead(odom_node["General"], "range_search_radius", range_search_radius_);
    YamlRead(odom_node["General"], "k_no_degrade", k_no_degrade_);
    YamlRead(odom_node["General"], "position_disable_thre_", position_disable_thre__);
    std::cout << "reprojection_cov:" << reprojection_cov << std::endl;
    std::cout << "ray_tracking_all_map:" << ray_tracking_all_map << std::endl;
    std::cout << "ray_tracking_range:[" << ray_tracking_range[0] << ", " << ray_tracking_range[1] << "]" << std::endl;
    std::cout << "range_search_radius_:" << range_search_radius_ << std::endl;
    std::cout << "k_no_degrade_:" << k_no_degrade_ << std::endl;
    std::cout << "position_disable_thre__:" << position_disable_thre__ << std::endl;

    tracking_manager_ptr = std::make_shared<slam_cam::TrackingManager>();
    tracker_ptr = std::make_shared<slam_cam::TrackerKLT>(tracking_manager_ptr, camera_ptr, num_feats, enhance_method,
    fast_threshold, grid_x, grid_y, min_px_dist);

    double max_dist_error, max_unconsistency;
    YamlRead(odom_node["General"], "landmark_use_dist_weight", landmark_use_dist_weight);
    YamlRead(odom_node["General"], "vio_use_dist_weight", vio_use_dist_weight);
    YamlRead(odom_node["General"], "vba_use_dist_weight", vba_use_dist_weight);
    YamlRead(odom_node["General"], "max_dist_error", max_dist_error);
    YamlRead(odom_node["General"], "max_unconsistency", max_unconsistency);
    std::cout << "landmark_use_dist_weight:" << landmark_use_dist_weight << ", vio_use_dist_weight:" << vio_use_dist_weight << ", vba_use_dist_weight:" << vba_use_dist_weight << std::endl;
    std::cout << "max_dist_error:" << max_dist_error << ", max_unconsistency:" << max_unconsistency << std::endl;
    if(landmark_use_dist_weight || vio_use_dist_weight || vba_use_dist_weight)
    {
      camera_ptr->ComputePixelWeight(max_dist_error, max_unconsistency);
    }
    
    YamlRead(odom_node["General"], "pub_dense", pub_dense_);
    YamlRead(odom_node["General"], "save_map_duration", save_map_duration_);
    std::cout << "pub_dense_:" << pub_dense_ << ", save_map_duration_:" << save_map_duration_ << std::endl;

    YamlRead(odom_node["General"], "record_memory", record_memory_);
    YamlRead(odom_node["General"], "malloc_trim_duration", malloc_trim_duration_);
    YamlRead(odom_node["General"], "max_jour", max_jour_);
    std::cout << "record_memory_:" << record_memory_ 
      << ", malloc_trim_duration_:" << malloc_trim_duration_ 
      << ", max_jour_:" << max_jour_ << std::endl;
    memory_log_ = std::ofstream(savepath+"/mem.csv");
    memory_log_ << "timestamp,mem"
      << ",id_tracking_map_,time_ids_map_,time_pose_cam_to_world_map_,time_pose_cam_to_world_mid_map_"
      << ",pair_count,octo_count,surf_map_mem"
      << ",pair_count,octo_count,surf_map_slide_mem"
      << ",sws_mem"
      << ",buf_lba2loop, buf_lba2loop_mem"
      << std::endl;

    YamlRead(odom_node["General"], "sleep_duration", sleep_duration_);
    std::cout << "sleep_duration_:" << sleep_duration_ << std::endl;

    std::string init_mode_str;
    YamlRead(odom_node["General"], "init_mode", init_mode_str);
    if(init_mode_str == "static")
      odom_ekf.init_mode = 1;
    else if(init_mode_str == "dynamic")
      odom_ekf.init_mode = 2;
    else
      odom_ekf.init_mode = 0;
    YamlRead(odom_node["General"], "gravity_align_en", odom_ekf.gravity_align_en);
    std::cout << "odom_ekf.init_mode:" << odom_ekf.init_mode << ", odom_ekf.gravity_align_en:" << odom_ekf.gravity_align_en << std::endl;

    YamlRead(odom_node["General"], "boundary_point_remove", boundary_point_remove_);
    if(boundary_point_remove_)
    {
      const std::string cfg_loam = cfg_path + "/RS_LOAM.yaml";
      YAML::Node loam_node = YAML::LoadFile(cfg_loam);
      lidar_odometry_.reset(new RsLidarOdometry(loam_node["lidar_odometry"]));
    }

    double cov_gyr, cov_acc, rand_walk_gyr, rand_walk_acc;
    YamlRead(odom_node["Odometry"], "max_iteration", odom_max_iter_);
    YamlRead(odom_node["Odometry"], "cov_gyr", cov_gyr);
    YamlRead(odom_node["Odometry"], "cov_acc", cov_acc);
    YamlRead(odom_node["Odometry"], "rdw_gyr", rand_walk_gyr);
    YamlRead(odom_node["Odometry"], "rdw_acc", rand_walk_acc);
    YamlRead(odom_node["Odometry"], "down_size", down_size);
    YamlRead(odom_node["Odometry"], "dept_err", dept_err);
    YamlRead(odom_node["Odometry"], "beam_err", beam_err);
    YamlRead(odom_node["Odometry"], "voxel_size", voxel_size);
    YamlRead(odom_node["Odometry"], "min_eigen_value", min_eigen_value);
    YamlRead(odom_node["Odometry"], "degrade_bound", degrade_bound);
    YamlRead(odom_node["Odometry"], "point_notime", point_notime);
    odom_ekf.point_notime = point_notime;
    std::cout << "degrade_bound:" << degrade_bound << std::endl;
    
    feat.blind = feat.blind * feat.blind;
    feat.blind_far = feat.blind_far * feat.blind_far;
    odom_ekf.cov_gyr << cov_gyr, cov_gyr, cov_gyr;
    odom_ekf.cov_acc << cov_acc, cov_acc, cov_acc;
    odom_ekf.cov_bias_gyr << rand_walk_gyr, rand_walk_gyr, rand_walk_gyr;
    odom_ekf.cov_bias_acc << rand_walk_acc, rand_walk_acc, rand_walk_acc;
    min_point << 5, 5, 5, 5;

    odom_ekf.rot_cam_to_imu_ = odom_ekf.Lid_rot_to_IMU * rot_lidar_to_cam_.transpose();
    odom_ekf.pos_cam_in_imu_ = odom_ekf.Lid_rot_to_IMU*rot_lidar_to_cam_.transpose()*(-pos_lidar_in_cam_) + odom_ekf.Lid_offset_to_IMU;
    std::cout << "rot_cam_to_imu_:\n" << odom_ekf.rot_cam_to_imu_ << std::endl;
    std::cout << "pos_cam_in_imu_:" << odom_ekf.pos_cam_in_imu_.transpose() << std::endl;

    YamlRead(odom_node["LocalBA"], "max_iteration", localBA_max_iter_);
    YamlRead(odom_node["LocalBA"], "win_size", win_size);
    YamlRead(odom_node["LocalBA"], "max_layer", max_layer);
    YamlRead(odom_node["LocalBA"], "cov_gyr", cov_gyr);
    YamlRead(odom_node["LocalBA"], "cov_acc", cov_acc);
    YamlRead(odom_node["LocalBA"], "rdw_gyr", rand_walk_gyr);
    YamlRead(odom_node["LocalBA"], "rdw_acc", rand_walk_acc);
    YamlRead(odom_node["LocalBA"], "min_ba_point", min_ba_point);
    YamlRead(odom_node["LocalBA"], "plane_eigen_value_thre", plane_eigen_value_thre);
    YamlRead(odom_node["LocalBA"], "imu_coef", imu_coef);
    YamlRead(odom_node["LocalBA"], "thread_num", thread_num);
    YamlRead(odom_node["LocalBA"], "enable", localBA_enable);
    YamlRead(odom_node["LocalBA"], "update_status", localBA_update_status);
    YamlRead(odom_node["LocalBA"], "reporjection_coeff", localBA_reprojection_coeff);
    YamlRead(odom_node["LocalBA"], "border_x", localBA_border_x);
    YamlRead(odom_node["LocalBA"], "border_y", localBA_border_y);
    localBA_border_x = std::max(0, localBA_border_x);
    localBA_border_y = std::max(0, localBA_border_y);
    YamlRead(odom_node["LocalBA"], "border_weight", localBA_border_weight);
    std::cout << "odom_max_iter_:" << odom_max_iter_ << ", localBA_max_iter_:" << localBA_max_iter_ << std::endl;

    of_landmark = std::ofstream(savepath + "/landmark.csv");
    of_landmark << "timestamp, lio_num, lidar_factor_num, tracking_num, cur_landmark_num_before, vio_num, visual_factor_num" << std::endl;
    time_cost_drop_ = std::ofstream(savepath + "/time_cost_drop.csv");
    time_cost_init_ = std::ofstream(savepath + "/time_cost_init.csv");
    time_cost_lvio_ = std::ofstream(savepath + "/time_cost_lvio.csv");
    time_cost_total_ = std::ofstream(savepath + "/time_cost_total.csv");
    time_cost_total_ << "name,total_time,start,end,spinOnce,x_curr.t" << std::endl;

    for(double &iter: plane_eigen_value_thre) iter = 1.0 / iter;

    noiseMeas.setZero(); noiseWalk.setZero();
    noiseMeas.diagonal() << cov_gyr, cov_gyr, cov_gyr, 
                            cov_acc, cov_acc, cov_acc;
    noiseWalk.diagonal() << 
    rand_walk_gyr, rand_walk_gyr, rand_walk_gyr, 
    rand_walk_acc, rand_walk_acc, rand_walk_acc;

    sws.resize(thread_num);
    cout << "bagname: " << bagname << endl;
  }

  ~VOXEL_SLAM()
  {
    if(!cloud_rgb_save.empty())
    {
      pcl::io::savePCDFileBinary(map_save_path_+ "/cloud_rgb_"+std::to_string(map_start_time_)+"_"+std::to_string(map_end_time_)+".pcd", cloud_rgb_save);
    }
    std::cout << "~VOXEL_SLAM()" << std::endl;
  }

  void LoadCalibration(const std::string cfg_path)
  {
    const std::string cfg_calib = cfg_path + "/calibration.yaml";
    YAML::Node calib_node = YAML::LoadFile(cfg_calib);
    if(!YamlNodeKeyExist(calib_node, "Sensor")) {std::cout << "key: Sensor not found in " << cfg_calib << std::endl; exit(1);}

    Eigen::Vector3d pos_IinB = Eigen::Vector3d::Zero();
    Eigen::Matrix3d rot_ItoB = Eigen::Matrix3d::Identity();
    {
      if(!YamlNodeKeyExist(calib_node["Sensor"], "IMU")) {std::cout << "key: Sensor:IMU not found in " << cfg_calib << std::endl; exit(1);}
      if(!YamlNodeKeyExist(calib_node["Sensor"]["IMU"], "extrinsic")) {std::cout << "key: Sensor:IMU:extrinsic not found in " << cfg_calib << std::endl; exit(1);}
      YAML::Node calib_node_imu = calib_node["Sensor"]["IMU"];
      YamlRead(calib_node_imu["extrinsic"]["translation"], "x", pos_IinB.x());
      YamlRead(calib_node_imu["extrinsic"]["translation"], "y", pos_IinB.y());
      YamlRead(calib_node_imu["extrinsic"]["translation"], "z", pos_IinB.z());
      Eigen::Quaterniond q_ItoB = Eigen::Quaterniond::Identity();
      YamlRead(calib_node_imu["extrinsic"]["quaternion"], "x", q_ItoB.x());
      YamlRead(calib_node_imu["extrinsic"]["quaternion"], "y", q_ItoB.y());
      YamlRead(calib_node_imu["extrinsic"]["quaternion"], "z", q_ItoB.z());
      YamlRead(calib_node_imu["extrinsic"]["quaternion"], "w", q_ItoB.w());
      rot_ItoB = q_ItoB.toRotationMatrix();
    }

    Eigen::Vector3d pos_LinB = Eigen::Vector3d::Zero();
    Eigen::Matrix3d rot_LtoB = Eigen::Matrix3d::Identity();
    {
      if(!YamlNodeKeyExist(calib_node["Sensor"], "Lidar")) {std::cout << "key: Sensor:Lidar not found in " << cfg_calib << std::endl; exit(1);}
      if(!YamlNodeKeyExist(calib_node["Sensor"]["Lidar"], "extrinsic")) {std::cout << "key: Sensor:Lidar:extrinsic not found in " << cfg_calib << std::endl; exit(1);}
      YAML::Node calib_node_lidar = calib_node["Sensor"]["Lidar"];
      YamlRead(calib_node_lidar["extrinsic"]["translation"], "x", pos_LinB.x());
      YamlRead(calib_node_lidar["extrinsic"]["translation"], "y", pos_LinB.y());
      YamlRead(calib_node_lidar["extrinsic"]["translation"], "z", pos_LinB.z());
      Eigen::Quaterniond q_LtoB = Eigen::Quaterniond::Identity();
      YamlRead(calib_node_lidar["extrinsic"]["quaternion"], "x", q_LtoB.x());
      YamlRead(calib_node_lidar["extrinsic"]["quaternion"], "y", q_LtoB.y());
      YamlRead(calib_node_lidar["extrinsic"]["quaternion"], "z", q_LtoB.z());
      YamlRead(calib_node_lidar["extrinsic"]["quaternion"], "w", q_LtoB.w());
      rot_LtoB = q_LtoB.toRotationMatrix();
    }

    Eigen::Vector3d pos_CinB = Eigen::Vector3d::Zero();
    Eigen::Matrix3d rot_CtoB = Eigen::Matrix3d::Identity();
    std::vector<double> proj_params, dist_params;
    std::vector<int> resolution;
    {
      if(!YamlNodeKeyExist(calib_node["Sensor"], "Camera")) {std::cout << "key: Sensor:Camera not found in " << cfg_calib << std::endl; exit(1);}
      if(!YamlNodeKeyExist(calib_node["Sensor"]["Camera"], "extrinsic")) {std::cout << "key: Sensor:Camera:extrinsic not found in " << cfg_calib << std::endl; exit(1);}
      if(!YamlNodeKeyExist(calib_node["Sensor"]["Camera"], "intrinsic")) {std::cout << "key: Sensor:Camera:intrinsic not found in " << cfg_calib << std::endl; exit(1);}
      YAML::Node calib_node_camera = calib_node["Sensor"]["Camera"];
      YamlRead(calib_node_camera["extrinsic"]["translation"], "x", pos_CinB.x());
      YamlRead(calib_node_camera["extrinsic"]["translation"], "y", pos_CinB.y());
      YamlRead(calib_node_camera["extrinsic"]["translation"], "z", pos_CinB.z());
      Eigen::Quaterniond q_CtoB = Eigen::Quaterniond::Identity();
      YamlRead(calib_node_camera["extrinsic"]["quaternion"], "x", q_CtoB.x());
      YamlRead(calib_node_camera["extrinsic"]["quaternion"], "y", q_CtoB.y());
      YamlRead(calib_node_camera["extrinsic"]["quaternion"], "z", q_CtoB.z());
      YamlRead(calib_node_camera["extrinsic"]["quaternion"], "w", q_CtoB.w());
      rot_CtoB = q_CtoB.toRotationMatrix();

      std::vector<double> proj_tmp;
      YamlRead(calib_node_camera["intrinsic"], "int_matrix", proj_tmp);
      proj_params.emplace_back(proj_tmp[0]);
      proj_params.emplace_back(proj_tmp[4]);
      proj_params.emplace_back(proj_tmp[2]);
      proj_params.emplace_back(proj_tmp[5]);
      if(proj_tmp[1]!=0.) proj_params.emplace_back(proj_tmp[1]);
      YamlRead(calib_node_camera["intrinsic"], "dist_coeff", dist_params);
      while(!dist_params.empty() && 0. == dist_params.back()) dist_params.pop_back();
      YamlRead(calib_node_camera["intrinsic"], "image_size", resolution);
    }

    odom_ekf.Lid_rot_to_IMU = rot_ItoB.transpose() * rot_LtoB;
    odom_ekf.Lid_offset_to_IMU = rot_ItoB.transpose() * (pos_LinB - pos_IinB);               
    extrin_para.R = odom_ekf.Lid_rot_to_IMU;
    extrin_para.p = odom_ekf.Lid_offset_to_IMU;
    rot_lidar_to_cam_ = rot_CtoB.transpose() * rot_LtoB;
    pos_lidar_in_cam_ = rot_CtoB.transpose() * (pos_LinB - pos_CinB);
    for(int i=0; i<3; i++)
    {
      if(std::abs(pos_lidar_in_cam_[i])<1e-4)
      {
        pos_lidar_in_cam_[i] = 0.;
      }
      else
      {
        pos_lidar_in_cam_[i] = std::round(10000*pos_lidar_in_cam_[i])/10000;
      }
    }

    std::cout << "rot_ItoB:\n" << rot_ItoB << std::endl;
    std::cout << "pos_IinB:" << pos_IinB.transpose() << std::endl;
    std::cout << "rot_LtoB:\n" << rot_LtoB << std::endl;
    std::cout << "pos_LinB:" << pos_LinB.transpose() << std::endl;
    std::cout << "rot_CtoB:\n" << rot_CtoB << std::endl;
    std::cout << "pos_CinB:" << pos_CinB.transpose() << std::endl;

    std::cout << "rot_LtoI:\n" << odom_ekf.Lid_rot_to_IMU << std::endl;
    std::cout << "pos_LinI:" << odom_ekf.Lid_offset_to_IMU.transpose() << std::endl;
    std::cout << "rot_LtoC:\n" << rot_lidar_to_cam_ << std::endl;
    std::cout << "pos_LinC:" << pos_lidar_in_cam_.transpose() << std::endl;
    std::cout << "resolution:" << resolution[0] << " " << resolution[1] << std::endl;
    std::cout << "proj_params:";
    for(double d:proj_params) std::cout << d << " ";
    std::cout << std::endl;
    std::cout << "dist_params:";
    for(double d:dist_params) std::cout << d << " ";
    std::cout << std::endl;

    const std::string cfg_odom = cfg_path + "/odom.yaml";
    YAML::Node odom_node = YAML::LoadFile(cfg_odom);

    std::string proj_model, dist_model;
    YamlRead(odom_node["General"], "projection_model", proj_model);
    YamlRead(odom_node["General"], "distortion_model", dist_model);
    YamlRead(odom_node["General"], "img_scale", img_scale);
    img_scale = std::max(0.1, std::min(1., img_scale));
    for(auto &tmp:resolution)
    {
      tmp = std::round(img_scale*tmp);
    }
    for(auto &tmp:proj_params)
    {
      tmp *= img_scale;
    }
    weight_scale_unit = 0.5 * (proj_params[0] + proj_params[1]);
    camera_ptr = slam_cam::CameraFactor::CreateCamera(resolution[0], resolution[1], proj_model, dist_model);
    camera_ptr->LoadParam(proj_params, dist_params);
    range_img_ = cv::Mat(resolution[1], resolution[0], CV_32FC1, cv::Scalar(0.f));
  }

  std::vector<size_t> getTotalMemory(const unordered_map<VOXEL_LOC, OctoTree*>& map) 
  {
    size_t count_pair{0}, count_octo{0};
    size_t total_size = sizeof(map);
    count_pair = map.size();
    total_size += (sizeof(VOXEL_LOC) + sizeof(OctoTree*)) * count_pair;
    // 每个键值对的大小
    for (const auto& pair : map) 
    {
      size_t count_ocot_tmp{0}, mem_running_size{0};
      pair.second->tras_size(count_ocot_tmp, mem_running_size);
      count_octo += count_ocot_tmp;
      total_size += sizeof(OctoTree) * count_ocot_tmp + mem_running_size;
    }
    return std::vector<size_t>({count_pair, count_octo, total_size});
  }


  bool vio_state_estimation(PVecPtr pptr, std::unordered_map<size_t, Eigen::Vector3d> &id_pw_map, std::vector<std::pair<size_t, std::pair<Eigen::Vector3d, Eigen::Vector2d>>> &id_bearing_uvs)
  {
    IMUST x_prop = x_curr;

    const int num_max_iter = odom_max_iter_;
    bool EKF_stop_flg = 0, flg_EKF_converged = 0;
    Eigen::Matrix<double, DIM, DIM> G, H_T_H, I_STATE;
    G.setZero(); H_T_H.setZero(); I_STATE.setIdentity();
    int rematch_num = 0;
    int match_num = 0;
    int match_num_vis = 0;

    int psize = pptr->size();
    std::vector<OctoTree*> octos;
    octos.resize(psize, nullptr);

    Eigen::Matrix3d nnt; 
    Eigen::Matrix<double, DIM, DIM> cov_inv = (x_curr.cov/reprojection_cov).inverse();

    double k_degrade = k_no_degrade_;
    if(lidar_degrade_)
    {
      k_degrade = 1.;
    }
    for(int iterCount=0; iterCount<num_max_iter; iterCount++)
    {
      double res_vis = 0.;
      Eigen::Matrix<double, 6, 6> HTH_vis; HTH_vis.setZero();
      Eigen::Matrix<double, 6, 1> HTz_vis; HTz_vis.setZero();
      Eigen::Matrix3d rot_cam_to_world = x_curr.R * odom_ekf.rot_cam_to_imu_;
      Eigen::Vector3d pos_cam_in_world = x_curr.R * odom_ekf.pos_cam_in_imu_ + x_curr.p;
      Eigen::Matrix3d rot_world_to_cam = rot_cam_to_world.transpose();
      Eigen::Vector3d pos_world_in_cam = rot_cam_to_world.transpose() * (-pos_cam_in_world);
      match_num_vis = 0;
      double R_vis_inv = 1./reprojection_cov;
      for(auto &id_bearing_uv:id_bearing_uvs)
      {
        if(id_pw_map.find(id_bearing_uv.first) == id_pw_map.end())
        {
          continue;
        }
        Eigen::Vector3d bearing = id_bearing_uv.second.first;
        Eigen::Vector2d uv_dist = id_bearing_uv.second.second;
        Eigen::Vector3d pw = id_pw_map.at(id_bearing_uv.first);
        Eigen::Vector3d pc = rot_world_to_cam * pw + pos_world_in_cam;
        if(camera_ptr->IsInBorder(uv_dist, border_x, border_y) && pc[2]>0.1)
        {
          Eigen::Vector2d pt = camera_ptr->BearingToPixel(pc);
          if(camera_ptr->IsInBorder(pt, border_x, border_y))
          {
            Eigen::Vector2d obs = bearing.head(2)/bearing[2];
            Eigen::Vector2d res = obs - pc.head(2) / pc(2);

            double weight = getHuberLossScale(res.norm() * weight_scale_unit,1);
            if(vio_use_dist_weight)
            {
              weight *= camera_ptr->GetPixelWeight(uv_dist);
            }
            weight *= k_degrade;
            Eigen::Matrix<double, 2, 6> jac;
            Eigen::Matrix<double, 2, 3> Jdt, JdR;
            Jdt << -1 / pc(2), 0,  pc(0) / (pc(2) * pc(2)) ,
                    0, -1 / pc(2), pc(1) / (pc(2) * pc(2));
            Jdt = Jdt * rot_cam_to_world.transpose();
            JdR <<  1 / pc(2), 0, - pc(0) / (pc(2) * pc(2)),
                    0, 1 / pc(2), - pc(1) / (pc(2) * pc(2));
            JdR = JdR * odom_ekf.rot_cam_to_imu_.transpose() * SkewSymmetric( x_curr.R.transpose() * (pw - x_curr.p));
            if(pc(2)>position_disable_thre__)
            {
              Jdt.setZero();
            }
            jac.block<2,3>(0,0) = JdR * weight * weight_scale_unit;
            jac.block<2,3>(0,3) = Jdt * weight * weight_scale_unit;

            Eigen::Vector2d z = res * weight * weight_scale_unit;
            HTH_vis += R_vis_inv * jac.transpose() * jac;
            HTz_vis += R_vis_inv * jac.transpose() * z;
            match_num_vis++;
            res_vis += z.transpose() * R_vis_inv * z;
          }
        }
      }
      Eigen::Matrix<double, 6, 6> HTH = HTH_vis; 
      Eigen::Matrix<double, 6, 1> HTz = HTz_vis; 
      if(match_num_vis<10)
      {
        break;
      }
      std::cout << "res_vis:" << res_vis << std::endl;

      H_T_H.block<6, 6>(0, 0) = HTH;
      Eigen::Matrix<double, DIM, DIM> K_1 = (H_T_H + cov_inv).inverse();
      G.block<DIM, 6>(0, 0) = K_1.block<DIM, 6>(0, 0) * HTH;
      Eigen::Matrix<double, DIM, 1> vec = x_prop - x_curr;
      Eigen::Matrix<double, DIM, 1> solution = K_1.block<DIM, 6>(0, 0) * HTz + vec - G.block<DIM, 6>(0, 0) * vec.block<6, 1>(0, 0);
      
      x_curr += solution;
      Eigen::Vector3d rot_add = solution.block<3, 1>(0, 0);
      Eigen::Vector3d tra_add = solution.block<3, 1>(3, 0);

      EKF_stop_flg = false;
      flg_EKF_converged = false;

      if ((rot_add.norm() * 57.3 < 0.001) && (tra_add.norm() * 100 < 0.001)) 
        flg_EKF_converged = true;

      if(flg_EKF_converged || (iterCount == num_max_iter-1))
      {
        x_curr.cov = (I_STATE - G) * x_curr.cov;
        EKF_stop_flg = true;
      }

      if(EKF_stop_flg) break;
    }

    landmark_nums[5] = match_num_vis;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(nnt);
    Eigen::Vector3d evalue = saes.eigenvalues();
    printf("eva %d, %d\n", match_num_vis, id_pw_map.size());
    Eigen::Matrix<double, DIM, 1> vec = x_prop - x_curr;
    std::cout << "vec_vis_delta:" << vec.transpose();
    return true;
  }

  // The point-to-plane alignment for odometry
  bool lio_state_estimation(PVecPtr pptr, std::unordered_map<size_t, Eigen::Vector3d> &id_pw_map, std::vector<std::pair<size_t, std::pair<Eigen::Vector3d, Eigen::Vector2d>>> &id_bearing_uvs)
  {
    IMUST x_prop = x_curr;

    const int num_max_iter = odom_max_iter_;
    bool EKF_stop_flg = 0, flg_EKF_converged = 0;
    Eigen::Matrix<double, DIM, DIM> G, H_T_H, I_STATE;
    G.setZero(); H_T_H.setZero(); I_STATE.setIdentity();
    int rematch_num = 0;
    int match_num = 0;
    int match_num_vis = 0;

    int psize = pptr->size();
    std::vector<OctoTree*> octos;
    octos.resize(psize, nullptr);

    Eigen::Matrix3d nnt; 
    Eigen::Matrix<double, DIM, DIM> cov_inv = x_curr.cov.inverse();
    for(int iterCount=0; iterCount<num_max_iter; iterCount++)
    {
      double res_lidar = 0., res_vis = 0.;
      Eigen::Matrix<double, 6, 6> HTH_lidar; HTH_lidar.setZero();
      Eigen::Matrix<double, 6, 1> HTz_lidar; HTz_lidar.setZero();
      Eigen::Matrix3d rot_var = x_curr.cov.block<3, 3>(0, 0);
      Eigen::Matrix3d tsl_var = x_curr.cov.block<3, 3>(3, 3);
      match_num = 0;
      nnt.setZero();
      latest_lio_octos.clear();

      for(int i=0; i<psize; i++)
      {
        pointVar &pv = pptr->at(i);
        Eigen::Matrix3d phat = hat(pv.pnt);
        Eigen::Matrix3d var_world = x_curr.R * pv.var * x_curr.R.transpose() + phat * rot_var * phat.transpose() + tsl_var;
        Eigen::Vector3d wld = x_curr.R * pv.pnt + x_curr.p;

        double sigma_d = 0;
        Plane* pla = nullptr;
        int flag = 0;
        if(octos[i] != nullptr && octos[i]->inside(wld))
        {
          double max_prob = 0;
          flag = octos[i]->match(wld, pla, max_prob, var_world, sigma_d, octos[i]);
        }
        else
        {
          flag = match(surf_map, wld, pla, var_world, sigma_d, octos[i]);
        }

        if(flag)
        {
          Plane &pp = *pla;
          double R_inv = 1.0 / (0.0005 + sigma_d);
          double resi = pp.normal.dot(wld - pp.center);

          Eigen::Matrix<double, 6, 1> jac;
          jac.head(3) = phat * x_curr.R.transpose() * pp.normal;
          jac.tail(3) = pp.normal;
          HTH_lidar += R_inv * jac * jac.transpose();
          HTz_lidar -= R_inv * jac * resi;
          nnt += pp.normal * pp.normal.transpose();
          match_num++;
          res_lidar += resi * R_inv * resi;

          latest_lio_octos.emplace_back(octos[i]);
        }
      }
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(nnt/match_num);
      Eigen::Vector3d evalue = saes.eigenvalues();
      lidar_degrade_ = evalue[0]<0.1;
      double k_degrade = k_no_degrade_;
      if(lidar_degrade_)
      {
        k_degrade = 1.;
      }
    

      Eigen::Matrix<double, 6, 6> HTH_vis; HTH_vis.setZero();
      Eigen::Matrix<double, 6, 1> HTz_vis; HTz_vis.setZero();
      Eigen::Matrix3d rot_cam_to_world = x_curr.R * odom_ekf.rot_cam_to_imu_;
      Eigen::Vector3d pos_cam_in_world = x_curr.R * odom_ekf.pos_cam_in_imu_ + x_curr.p;
      Eigen::Matrix3d rot_world_to_cam = rot_cam_to_world.transpose();
      Eigen::Vector3d pos_world_in_cam = rot_cam_to_world.transpose() * (-pos_cam_in_world);
      match_num_vis = 0;
      double R_vis_inv = 1./reprojection_cov;
      for(auto &id_bearing_uv:id_bearing_uvs)
      {
        if(id_pw_map.find(id_bearing_uv.first) == id_pw_map.end())
        {
          continue;
        }
        Eigen::Vector3d bearing = id_bearing_uv.second.first;
        Eigen::Vector2d uv_dist = id_bearing_uv.second.second;
        Eigen::Vector3d pw = id_pw_map.at(id_bearing_uv.first);
        Eigen::Vector3d pc = rot_world_to_cam * pw + pos_world_in_cam;
        if(camera_ptr->IsInBorder(uv_dist, border_x, border_y) && pc[2]>0.1)
        {
          Eigen::Vector2d pt = camera_ptr->BearingToPixel(pc);
          if(camera_ptr->IsInBorder(pt, border_x, border_y))
          {
            Eigen::Vector2d obs = bearing.head(2)/bearing[2];
            Eigen::Vector2d res = obs - pc.head(2) / pc(2);

            double weight = getHuberLossScale(res.norm() * weight_scale_unit,1);
            if(vio_use_dist_weight)
            {
              weight *= camera_ptr->GetPixelWeight(uv_dist);
            }
            weight *= k_degrade;
            Eigen::Matrix<double, 2, 6> jac;
            Eigen::Matrix<double, 2, 3> Jdt, JdR;
            Jdt << -1 / pc(2), 0,  pc(0) / (pc(2) * pc(2)) ,
                    0, -1 / pc(2), pc(1) / (pc(2) * pc(2));
            Jdt = Jdt * rot_cam_to_world.transpose();
            JdR <<  1 / pc(2), 0, - pc(0) / (pc(2) * pc(2)),
                    0, 1 / pc(2), - pc(1) / (pc(2) * pc(2));
            JdR = JdR * odom_ekf.rot_cam_to_imu_.transpose() * SkewSymmetric( x_curr.R.transpose() * (pw - x_curr.p));
            if(pc(2)>position_disable_thre__)
            {
              Jdt.setZero();
            }
            jac.block<2,3>(0,0) = JdR * weight * weight_scale_unit;
            jac.block<2,3>(0,3) = Jdt * weight * weight_scale_unit;

            Eigen::Vector2d z = res * weight * weight_scale_unit;
            HTH_vis += R_vis_inv* jac.transpose() * jac;
            HTz_vis += R_vis_inv* jac.transpose() * z;
            match_num_vis++;
            res_vis += z.transpose() * R_vis_inv * z;
          }
        }
      }
      
      Eigen::Matrix<double, 6, 6> HTH = HTH_lidar; 
      Eigen::Matrix<double, 6, 1> HTz = HTz_lidar; 
      if(match_num_vis>10)
      {
        if(re_weight)
        {
          double k = match_num / match_num_vis;
          HTH_vis *= k;
          HTz_vis *= k;
        }
        HTH += HTH_vis;
        HTz += HTz_vis;
        std::cout << "res_lidar:" << res_lidar << ", res_vis:" << res_vis << std::endl;
      }
      else
      {
        std::cout << "res_lidar:" << res_lidar << std::endl;
      }

      std::cout << "match_num:" << match_num << ", match_num_vis:" << match_num_vis << ", R_vis_inv:" << R_vis_inv << std::endl;

      H_T_H.block<6, 6>(0, 0) = HTH;
      Eigen::Matrix<double, DIM, DIM> K_1 = (H_T_H + cov_inv).inverse();
      G.block<DIM, 6>(0, 0) = K_1.block<DIM, 6>(0, 0) * HTH;
      Eigen::Matrix<double, DIM, 1> vec = x_prop - x_curr;
      Eigen::Matrix<double, DIM, 1> solution = K_1.block<DIM, 6>(0, 0) * HTz + vec - G.block<DIM, 6>(0, 0) * vec.block<6, 1>(0, 0);
      
      x_curr += solution;
      Eigen::Vector3d rot_add = solution.block<3, 1>(0, 0);
      Eigen::Vector3d tra_add = solution.block<3, 1>(3, 0);

      EKF_stop_flg = false;
      flg_EKF_converged = false;

      if ((rot_add.norm() * 57.3 < 0.01) && (tra_add.norm() * 100 < 0.015)) 
        flg_EKF_converged = true;

      if(flg_EKF_converged || ((rematch_num==0) && (iterCount==num_max_iter-2)))
      {       
        rematch_num++;
      }

      if(rematch_num >= 2 || (iterCount == num_max_iter-1))
      {
        x_curr.cov = (I_STATE - G) * x_curr.cov;
        EKF_stop_flg = true;
      }

      if(EKF_stop_flg) break;
    }

    landmark_nums[1] = match_num;
    landmark_nums[5] = match_num_vis;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(nnt/match_num);
    Eigen::Vector3d evalue = saes.eigenvalues();
    printf("eva %d: %lf, %d, %d, %f\n", match_num, evalue[0], match_num_vis, id_pw_map.size(), (re_weight && (match_num_vis>10)) ? (match_num / match_num_vis):0.f);
    static std::ofstream of(savepath + "/evalue.txt", std::ios::out);
    of << std::to_string(x_curr.t) << " " << evalue[0] << " " << evalue[1] << " " << evalue[2] << " 0 0 0 1" << std::endl;

    if(evalue[0] < 0.1)
      return false;
    else
      return true;
  }

  // The point-to-plane alignment for initialization
  pcl::PointCloud<PointType>::Ptr pl_tree;
  void lio_state_estimation_kdtree(PVecPtr pptr)
  {
    static pcl::KdTreeFLANN<PointType> kd_map;
    if(pl_tree->size() < 100)
    {
      for(pointVar pv: *pptr)
      {
        PointType pp;
        pv.pnt = x_curr.R * pv.pnt + x_curr.p;
        pp.x = pv.pnt[0]; pp.y = pv.pnt[1]; pp.z = pv.pnt[2];
        pl_tree->push_back(pp);
      }
      kd_map.setInputCloud(pl_tree);
      return;
    }

    const int num_max_iter = 4;
    IMUST x_prop = x_curr;
    int psize = pptr->size();
    bool EKF_stop_flg = 0, flg_EKF_converged = 0;
    Eigen::Matrix<double, DIM, DIM> G, H_T_H, I_STATE;
    G.setZero(); H_T_H.setZero(); I_STATE.setIdentity();

    double max_dis = 2*2;
    std::vector<float> sqdis(NMATCH); std::vector<int> nearInd(NMATCH);
    PLV(3) vecs(NMATCH);
    int rematch_num = 0;
    Eigen::Matrix<double, DIM, DIM> cov_inv = x_curr.cov.inverse();

    Eigen::Matrix<double, NMATCH, 1> b;
    b.setOnes();
    b *= -1.0f;

    std::vector<double> ds(psize, -1);
    PLV(3) directs(psize);
    bool refind = true;

    for(int iterCount=0; iterCount<num_max_iter; iterCount++)
    {
      Eigen::Matrix<double, 6, 6> HTH; HTH.setZero();
      Eigen::Matrix<double, 6, 1> HTz; HTz.setZero();
      int valid = 0;
      for(int i=0; i<psize; i++)
      {
        pointVar &pv = pptr->at(i);
        Eigen::Matrix3d phat = hat(pv.pnt);
        Eigen::Vector3d wld = x_curr.R * pv.pnt + x_curr.p;

        if(refind)
        {
          PointType apx;
          apx.x = wld[0]; apx.y = wld[1]; apx.z = wld[2];
          kd_map.nearestKSearch(apx, NMATCH, nearInd, sqdis);

          Eigen::Matrix<double, NMATCH, 3> A;
          for(int i=0; i<NMATCH; i++)
          {
            PointType &pp = pl_tree->points[nearInd[i]];
            A.row(i) << pp.x, pp.y, pp.z;
          }
          Eigen::Vector3d direct = A.colPivHouseholderQr().solve(b);
          bool check_flag = false;
          for(int i=0; i<NMATCH; i++)
          {
            if(fabs(direct.dot(A.row(i)) + 1.0) > 0.1) 
              check_flag = true;
          }

          if(check_flag) 
          {
            ds[i] = -1;
            continue;
          }
          
          double d = 1.0 / direct.norm();
          ds[i] = d;
          directs[i] = direct * d;
        }

        if(ds[i] >= 0)
        {
          double pd2 = directs[i].dot(wld) + ds[i];
          Eigen::Matrix<double, 6, 1> jac_s;
          jac_s.head(3) = phat * x_curr.R.transpose() * directs[i];
          jac_s.tail(3) = directs[i];

          HTH += jac_s * jac_s.transpose();
          HTz += jac_s * (-pd2);
          valid++;
        }
      }

      H_T_H.block<6, 6>(0, 0) = HTH;
      Eigen::Matrix<double, DIM, DIM> K_1 = (H_T_H + cov_inv / 1000).inverse();
      G.block<DIM, 6>(0, 0) = K_1.block<DIM, 6>(0, 0) * HTH;
      Eigen::Matrix<double, DIM, 1> vec = x_prop - x_curr;
      Eigen::Matrix<double, DIM, 1> solution = K_1.block<DIM, 6>(0, 0) * HTz + vec - G.block<DIM, 6>(0, 0) * vec.block<6, 1>(0, 0);

      x_curr += solution;
      Eigen::Vector3d rot_add = solution.block<3, 1>(0, 0);
      Eigen::Vector3d tra_add = solution.block<3, 1>(3, 0);

      refind = false;
      if ((rot_add.norm() * 57.3 < 0.01) && (tra_add.norm() * 100 < 0.015))
      {
        refind = true;
        flg_EKF_converged = true;
        rematch_num++;
      }

      if(iterCount == num_max_iter-2 && !flg_EKF_converged)
      {
        refind = true;
      }

      if(rematch_num >= 2 || (iterCount == num_max_iter-1))
      {
        x_curr.cov = (I_STATE - G) * x_curr.cov;
        EKF_stop_flg = true;
      }

      if(EKF_stop_flg) break;
    }

    auto tt1 = std::chrono::high_resolution_clock::now();
    for(pointVar pv: *pptr)
    {
      pv.pnt = x_curr.R * pv.pnt + x_curr.p;
      PointType ap;
      ap.x = pv.pnt[0]; ap.y = pv.pnt[1]; ap.z = pv.pnt[2];
      pl_tree->push_back(ap);
    }
    down_sampling_voxel(*pl_tree, 0.5);
    kd_map.setInputCloud(pl_tree);
    auto tt2 = std::chrono::high_resolution_clock::now();
  }

  int initialization(deque<RosImuPtr> &imus, Eigen::MatrixXd &hess, LidarFactor &voxhess, PLV(3) &pwld, pcl::PointCloud<PointType>::Ptr pcl_curr)
  {
    static std::vector<pcl::PointCloud<PointType>::Ptr> pl_origs;
    static std::vector<double> beg_times;
    static std::vector<deque<RosImuPtr>> vec_imus;

    pcl::PointCloud<PointType>::Ptr orig(new pcl::PointCloud<PointType>(*pcl_curr));
    if(odom_ekf.process(x_curr, *pcl_curr, imus) == 0)
      return 0;
    if(win_count == 0)
      imupre_scale_gravity = odom_ekf.scale_gravity;

    PVecPtr pptr(new PVec);
    double downkd = down_size >= 0.5 ? down_size : 0.5;
    down_sampling_voxel(*pcl_curr, downkd);
    var_init(extrin_para, *pcl_curr, pptr, dept_err, beam_err);
    lio_state_estimation_kdtree(pptr);
    if(1==odom_ekf.init_mode)
    {
      x_curr.R = odom_ekf.x_first.R;
      x_curr.p = odom_ekf.x_first.p;
      x_curr.v = odom_ekf.x_first.v;
      x_curr.bg = odom_ekf.x_first.bg;
      x_curr.ba = odom_ekf.x_first.ba;
      x_curr.g = odom_ekf.x_first.g;
    }

    pwld.clear();
    pvec_update(pptr, x_curr, pwld);

    win_count++;
    x_buf.push_back(x_curr);
    pvec_buf.push_back(pptr);
    ResultOutput::instance().pub_localtraj(pwld, 0, x_curr, sessionNames.size()-1, pcl_path, pptr);

    if(win_count > 1)
    {
      imu_pre_buf.push_back(new IMU_PRE(x_buf[win_count-2].bg, x_buf[win_count-2].ba));
      imu_pre_buf[win_count-2]->push_imu(imus);
    }
    pcl::PointCloud<PointType> pl_mid = *orig;
    down_sampling_close(*orig, down_size);
    if(orig->size() < 1000)
    {
      *orig = pl_mid;
      down_sampling_close(*orig, down_size / 2);
    }
    sort(orig->begin(), orig->end(), [](PointType &x, PointType &y)
    {return x.curvature < y.curvature;});

    pl_origs.push_back(orig);
    beg_times.push_back(odom_ekf.pcl_beg_time);
    vec_imus.push_back(imus);

    int is_success = 0;
    if(win_count >= win_size)
    {
      is_success = Initialization::instance().motion_init(pl_origs, vec_imus, beg_times, &hess, voxhess, x_buf, surf_map, surf_map_slide, pvec_buf, win_size, sws, x_curr, imu_pre_buf, extrin_para, odom_ekf.init_mode);

      if(is_success == 0)
      {
        return -1;
      }
      return 1;
    }
    return 0;
  }

  void system_reset(deque<RosImuPtr> &imus)
  {
    for(auto iter=surf_map.begin(); iter!=surf_map.end(); iter++)
    {
      iter->second->tras_ptr(octos_release);
      iter->second->clear_slwd(sws[0]);
      delete iter->second;
    }
    surf_map.clear(); surf_map_slide.clear();

    x_curr.setZero();
    x_curr.p = Eigen::Vector3d(0, 0, 30);
    odom_ekf.mean_acc.setZero();
    odom_ekf.init_num = 0;
    odom_ekf.IMU_init(imus);
    odom_ekf.x_prev.t = -1.;
    x_curr.g = -odom_ekf.mean_acc * imupre_scale_gravity;

    for(int i=0; i<imu_pre_buf.size(); i++)
      delete imu_pre_buf[i];
    x_buf.clear(); pvec_buf.clear(); imu_pre_buf.clear();
    pl_tree->clear();

    for(int i=0; i<win_size; i++)
      mp[i] = i;
    win_base = 0; win_count = 0; pcl_path.clear();
    pub_pl_func(pcl_path, pub_cmap);
    std::cout << "Reset!!!" << std::endl;
  }

  // After local BA, update the map and marginalize the points of oldest scan
  // multi means multiple thread
  void multi_margi(unordered_map<VOXEL_LOC, OctoTree*> &feat_map, double jour, int win_count, std::vector<IMUST> &xs, LidarFactor &voxopt, std::vector<SlideWindow*> &sw)
  {
    int thd_num = thread_num;
    std::vector<vector<OctoTree*>*> octs;
    for(int i=0; i<thd_num; i++) 
      octs.push_back(new std::vector<OctoTree*>());

    int g_size = feat_map.size();
    if(g_size < thd_num) return;
    std::vector<thread*> mthreads(thd_num);
    double part = 1.0 * g_size / thd_num;
    int cnt = 0;
    for(auto iter=feat_map.begin(); iter!=feat_map.end(); iter++)
    {
      iter->second->jour = jour;
      octs[cnt]->push_back(iter->second);
      if(octs[cnt]->size() >= part && cnt < thd_num-1)
        cnt++;
    }

    auto margi_func = [](int win_cnt, std::vector<OctoTree*> *oct, std::vector<IMUST> xxs, LidarFactor &voxhess)
    {
      for(OctoTree *oc: *oct)
      {
        oc->margi(win_cnt, 1, xxs, voxhess);
      }
    };

    for(int i=1; i<thd_num; i++)
    {
      mthreads[i] = new thread(margi_func, win_count, octs[i], xs, ref(voxopt));
    }
    
    for(int i=0; i<thd_num; i++)
    {
      if(i == 0)
      {
        margi_func(win_count, octs[i], xs, voxopt);
      }
      else
      {
        mthreads[i]->join();
        delete mthreads[i];
      }
    }

    for(auto iter=feat_map.begin(); iter!=feat_map.end();)
    {
      if(iter->second->isexist)
        iter++;
      else
      {
        iter->second->clear_slwd(sw);
        feat_map.erase(iter++);
      }
    }

    for(int i=0; i<thd_num; i++)
      delete octs[i];

  }

  // Determine the plane and recut the voxel map in octo-tree
  void multi_recut(unordered_map<VOXEL_LOC, OctoTree*> &feat_map, int win_count, std::vector<IMUST> &xs, LidarFactor &voxopt, std::vector<vector<SlideWindow*>> &sws)
  {

    int thd_num = thread_num;
    std::vector<vector<OctoTree*>> octss(thd_num);
    int g_size = feat_map.size();
    if(g_size < thd_num) return;
    std::vector<thread*> mthreads(thd_num);
    double part = 1.0 * g_size / thd_num;
    int cnt = 0;
    for(auto iter=feat_map.begin(); iter!=feat_map.end(); iter++)
    {
      octss[cnt].push_back(iter->second);
      if(octss[cnt].size() >= part && cnt < thd_num-1)
        cnt++;
    }

    auto recut_func = [](int win_count, std::vector<OctoTree*> &oct, std::vector<IMUST> xxs, std::vector<SlideWindow*> &sw)
    {
      for(OctoTree *oc: oct)
        oc->recut(win_count, xxs, sw);
    };

    for(int i=1; i<thd_num; i++)
    {
      mthreads[i] = new thread(recut_func, win_count, ref(octss[i]), xs, ref(sws[i]));
    }

    for(int i=0; i<thd_num; i++)
    {
      if(i == 0)
      {
        recut_func(win_count, octss[i], xs, sws[i]);
      }
      else
      {
        mthreads[i]->join();
        delete mthreads[i];
      }
    }

    for(int i=1; i<sws.size(); i++)
    {
      sws[0].insert(sws[0].end(), sws[i].begin(), sws[i].end());
      sws[i].clear();
    }

    for(auto iter=feat_map.begin(); iter!=feat_map.end(); iter++)
      iter->second->tras_opt(voxopt);

  }

  // The main thread of odometry and local mapping
  void thd_odometry_localmapping()
  {
    std::chrono::_V2::system_clock::time_point t0, t1, start, end;
    std::vector<std::string> col_key;
    std::vector<double> col_val; // [ms]
    bool drop_first{true}, init_first{true}, lvio_first{true};
    bool is_init{false};

    PLV(3) pwld;
    PLV(3) pwld_pub;
    double down_sizes[3] = {0.1, 0.2, 0.4};
    Eigen::Vector3d last_pos(0, 0 ,0);
    double jour = 0;
    int counter = 0;

    pcl::PointCloud<PointType>::Ptr pcl_curr(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr pcl_pub(new pcl::PointCloud<PointType>());
    int motion_init_flag = 1;
    pl_tree.reset(new pcl::PointCloud<PointType>());
    std::vector<pcl::PointCloud<PointType>::Ptr> pl_origs;
    std::vector<double> beg_times;
    std::vector<deque<RosImuPtr>> vec_imus;
    bool release_flag = false;
    int degrade_cnt = 0;
    LidarFactor voxhess(win_size);
    VisualFactorParam visual_factor_param;
    visual_factor_param.rot_cam_to_imu = odom_ekf.rot_cam_to_imu_;
    visual_factor_param.pos_cam_in_imu = odom_ekf.pos_cam_in_imu_;
    visual_factor_param.weight_scale_unit = weight_scale_unit;
    visual_factor_param.use_kernel = true;
    visual_factor_param.loss_function = getHuberLossScale;
    visual_factor_param.kernel_delta = 1.;
    visual_factor_param.reporjection_coeff = localBA_reprojection_coeff;
    visual_factor_param.camera_ptr = camera_ptr;
    visual_factor_param.border_x = localBA_border_x;
    visual_factor_param.border_y = localBA_border_y;
    visual_factor_param.border_weight = localBA_border_weight;
    visual_factor_param.use_dist_weight = vba_use_dist_weight;

    VisualFactor visual_factor(win_size, visual_factor_param);
    const int mgsize = 1;
    Eigen::MatrixXd hess;
    int malloc_trim_cnt = 0;
    while(odom_alive_)
    {
      col_key.clear();
      col_val.clear();
      is_init = false;
      start = std::chrono::high_resolution_clock::now();

      std::deque<RosImuPtr> imus;

      t0 = std::chrono::high_resolution_clock::now();
      bool sync_ok{false};
      if(0 == sync_mode)
      {
        sync_ok = sync_packages(pcl_curr, imus, odom_ekf);
      }
      else
      {
        sync_ok = sync_packages_new(pcl_curr, imus, odom_ekf);
      }
      if(!sync_ok)
      {
        t1 = std::chrono::high_resolution_clock::now();
        col_key.push_back("sync_packages");
        col_val.push_back(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()*1e-3);

        malloc_trim_cnt = 0;

        t0 = std::chrono::high_resolution_clock::now();
        if(octos_release.size() != 0)
        {
          int msize = octos_release.size();
          if(msize > 1000) msize = 1000;
          for(int i=0; i<msize; i++)
          {
            delete octos_release.back();
            octos_release.pop_back();
          }
          malloc_trim(0);
        }
        else if(release_flag && max_jour_>=0.)
        {
          release_flag = false;
          std::vector<OctoTree*> octos;
          for(auto iter=surf_map.begin(); iter!=surf_map.end();)
          {
            int dis = jour - iter->second->jour;
            if(dis < max_jour_)
            {
              iter++;
            }
            else
            {
              octos.push_back(iter->second);
              iter->second->tras_ptr(octos);
              surf_map.erase(iter++);
            }
          }
          int ocsize = octos.size();
          for(int i=0; i<ocsize; i++)
            delete octos[i];
          octos.clear();
          malloc_trim(0);
        }
        else if(sws[0].size() > 10000)
        {
          for(int i=0; i<500; i++)
          {
            delete sws[0].back();
            sws[0].pop_back();
          }
          malloc_trim(0);
        }
        t1 = std::chrono::high_resolution_clock::now();
        end = std::chrono::high_resolution_clock::now();
        col_key.push_back("release_map");
        col_val.push_back(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()*1e-3);
        col_key.push_back("drop_total");
        col_val.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()*1e-3);
        if(drop_first)
        {
          drop_first = false;
          for(std::string key:col_key)
          {
            time_cost_drop_ << key << ",";
          }
          time_cost_drop_ << std::endl;
        }
        for(double val:col_val)
        {
          time_cost_drop_ << val << ",";
        }
        time_cost_drop_ << std::endl;
        time_cost_total_ << "drop," << col_val.back() 
          << "," << std::to_string(std::chrono::duration<double>(start.time_since_epoch()).count())
          << "," << std::to_string(std::chrono::duration<double>(end.time_since_epoch()).count())
          << "," << col_val.front()
          << std::endl;
        if(sleep_duration_>0)
        {
          std::this_thread::sleep_for(std::chrono::milliseconds(sleep_duration_));
        }
        continue;
      }
      t1 = std::chrono::high_resolution_clock::now();
      col_key.push_back("sync_packages");
      col_val.push_back(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()*1e-3);

      t0 = std::chrono::high_resolution_clock::now();
      malloc_trim_cnt++;
      if(malloc_trim_duration_>0 && malloc_trim_cnt>malloc_trim_duration_)
      {
        malloc_trim(0);
        malloc_trim_cnt = 1;
      }
      t1 = std::chrono::high_resolution_clock::now();
      col_key.push_back("release_data_buffer");
      col_val.push_back(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()*1e-3);

      t0 = std::chrono::high_resolution_clock::now();
      static int first_flag = 1;
      if (first_flag)
      {
        pcl::PointCloud<PointType> pl;
        pub_pl_func(pl, pub_pmap);
        pub_pl_func(pl, pub_prev_path);
        first_flag = 0;
      }
      t1 = std::chrono::high_resolution_clock::now();
      col_key.push_back("first_flag");
      col_val.push_back(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()*1e-3);
      
      if(motion_init_flag)
      {
        is_init = true;
        t0 = std::chrono::high_resolution_clock::now();
        int init = initialization(imus, hess, voxhess, pwld, pcl_curr);
        t1 = std::chrono::high_resolution_clock::now();
        col_key.push_back("init");
        col_val.push_back(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()*1e-3);
        if(init == 1)
        {
          motion_init_flag = 0;
        }
        else
        {
          if(init == -1)
            system_reset(imus);
          continue;
        }
      }
      else
      {
        is_init = false;
        t0 = std::chrono::high_resolution_clock::now();
        if(odom_ekf.process(x_curr, *pcl_curr, imus) == 0)
          continue;
        t1 = std::chrono::high_resolution_clock::now();
        col_key.push_back("undistort");
        col_val.push_back(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()*1e-3);

        if(boundary_point_remove_)
        {
          for (auto &p:pcl_curr->points) 
          {
            p.curvature = 99;
          }
          lidar_odometry_->AddLiDAR(pcl_curr, x_curr.t);
          *pcl_curr = lidar_odometry_->GetValidCloud();
        }
        if(pcl_curr->size()<100)
        {
          continue;
        }

        t0 = std::chrono::high_resolution_clock::now();
        landmark_nums[0] = x_curr.t;
        pcl::PointCloud<PointType> pl_down = *pcl_curr;        
        down_sampling_voxel(pl_down, down_size);

        if(pl_down.size() < 500)
        {
          pl_down = *pcl_curr;
          down_sampling_voxel(pl_down, down_size / 2);
        }
        t1 = std::chrono::high_resolution_clock::now();
        col_key.push_back("downsample");
        col_val.push_back(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()*1e-3);

        if(pub_dense_)
        {
          *pcl_pub = *pcl_curr;
        }
        else
        {
          *pcl_pub = pl_down;
        }

        t0 = std::chrono::high_resolution_clock::now();
        PVecPtr pptr(new PVec);
        PVecPtr pptr_pub(new PVec);
        var_init(extrin_para, pl_down, pptr, dept_err, beam_err);
        var_init(extrin_para, *pcl_pub, pptr_pub);
        t1 = std::chrono::high_resolution_clock::now();
        col_key.push_back("var_init");
        col_val.push_back(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()*1e-3);

        t0 = std::chrono::high_resolution_clock::now();
        if(odom_ekf.img_match_status == 1)
        {
          range_img_.setTo(0.f);
          int count{0};
          for(auto &p3d:pcl_curr->points)
          {
            Eigen::Vector3d pl(p3d.x, p3d.y, p3d.z);
            Eigen::Vector3d pc = rot_lidar_to_cam_ * pl + pos_lidar_in_cam_;
            if(pc[2]>0.01)
            {
              Eigen::Vector2d pt = camera_ptr->BearingToPixel(pc);
              if(camera_ptr->IsInBorder(pt, 0, 0))
              {
                if(0.f == range_img_.at<float>(pt[1], pt[0]))
                {
                  range_img_.at<float>(pt[1], pt[0]) = pc.norm();
                  count++;
                }
                else if(pc[2] < range_img_.at<float>(pt[1], pt[0]))
                {
                  range_img_.at<float>(pt[1], pt[0]) = pc.norm();
                }
              }
            }
          }
          if(range_search_radius_ < 0)
          {
            int max_radius = 2*std::ceil(std::sqrt(double(range_img_.rows * range_img_.cols) / count));
            cv::Mat bin_img;
            range_img_.convertTo(bin_img, CV_8UC1, 100*255, 50);
            cv::imwrite(radius_save_path_+"/"+std::to_string(0)+".jpg", bin_img);
            int dilation_size = 1;
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,
                                               cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
                                               cv::Point(-1, -1));

            for(int r=1; r<=max_radius; r++)
            {
              cv::dilate(bin_img, bin_img, kernel);
              cv::imwrite(radius_save_path_+"/"+std::to_string(r)+".jpg", bin_img);
            }
            std::cout << "radius imgs has been saved to " << radius_save_path_ << ", please select the best radius manually and set it to range_search_radius" << std::endl;
            exit(1);
          }
        }
        t1 = std::chrono::high_resolution_clock::now();
        col_key.push_back("range_img");
        col_val.push_back(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()*1e-3);

        t0 = std::chrono::high_resolution_clock::now();
        std::unordered_map<size_t, Eigen::Vector3d> id_pw_map_landmark;
        std::vector<std::pair<size_t, std::pair<Eigen::Vector3d, Eigen::Vector2d>>> id_bearing_uvs_landmark;
        tracking_ptrs_curr.clear();
        if(odom_ekf.img_match_status == 1)
        {
          while(last_img_time < odom_ekf.img_match_time)
          {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
          }
          if(use_landmark)
          {
            tracking_manager_ptr->GetTrackingPtrs(odom_ekf.img_match_time, tracking_ptrs_curr);
            GetVisualFactorByOdom(tracking_ptrs_curr, odom_ekf.img_match_time, id_pw_map_landmark, id_bearing_uvs_landmark);
            std::cout << "tracking_ptrs_curr.size():" << tracking_ptrs_curr.size() << ", id_pw_map_landmark.size():" << id_pw_map_landmark.size() << ", id_bearing_uvs_landmark.size():" << id_bearing_uvs_landmark.size() << std::endl;
            landmark_nums[3] = tracking_ptrs_curr.size();
            landmark_nums[4] = id_pw_map_landmark.size();
          }
        }
        else
        {
          landmark_nums[3] = 0;
          landmark_nums[4] = 0;
        }
        t1 = std::chrono::high_resolution_clock::now();
        col_key.push_back("vio_pre");
        col_val.push_back(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()*1e-3);

        t0 = std::chrono::high_resolution_clock::now();
        switch(vis_update_status)
        {
          case 0:
          {
            std::unordered_map<size_t, Eigen::Vector3d> id_pw_map_empty;
            std::vector<std::pair<size_t, std::pair<Eigen::Vector3d, Eigen::Vector2d>>> id_bearing_uvs_empty;
            if(lio_state_estimation(pptr, id_pw_map_empty, id_bearing_uvs_empty))
            {
              if(degrade_cnt > 0) degrade_cnt--;
            }
            else
              degrade_cnt++;
          }
          break;
          case 1:
          {
            std::unordered_map<size_t, Eigen::Vector3d> id_pw_map_empty;
            std::vector<std::pair<size_t, std::pair<Eigen::Vector3d, Eigen::Vector2d>>> id_bearings_empty;
            if(lio_state_estimation(pptr, id_pw_map_empty, id_bearings_empty))
            {
              if(degrade_cnt > 0) degrade_cnt--;
            }
            else
              degrade_cnt++;
            if(use_landmark)
            {
              vio_state_estimation(pptr, id_pw_map_landmark, id_bearing_uvs_landmark);
            }
          }
          break;
          case 2:
          {
            if(use_landmark)
            {
              if(lio_state_estimation(pptr, id_pw_map_landmark, id_bearing_uvs_landmark))
              {
                if(degrade_cnt > 0) degrade_cnt--;
              }
              else
                degrade_cnt++;
            }
          }
          break;
          default:
          {
            std::cout << "vis_update_status:" << vis_update_status << " is not defined." << std::endl;
            std::unordered_map<size_t, Eigen::Vector3d> id_pw_map_empty;
            std::vector<std::pair<size_t, std::pair<Eigen::Vector3d, Eigen::Vector2d>>> id_bearing_uvs_empty;
            if(lio_state_estimation(pptr, id_pw_map_empty, id_bearing_uvs_empty))
            {
              if(degrade_cnt > 0) degrade_cnt--;
            }
            else
              degrade_cnt++;
          }
          break;
        }
        t1 = std::chrono::high_resolution_clock::now();
        col_key.push_back("lvio");
        col_val.push_back(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()*1e-3);

        t0 = std::chrono::high_resolution_clock::now();
        if(!ray_tracking_all_map)
        {
          for(OctoTree* octo:latest_lio_octos)
          {
            octo->latest_lio = true;
          }
        }
        t1 = std::chrono::high_resolution_clock::now();
        col_key.push_back("latest_lio");
        col_val.push_back(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()*1e-3);
        
        t0 = std::chrono::high_resolution_clock::now();
        pwld.clear();
        pvec_update(pptr, x_curr, pwld);
        pwld_pub.clear();
        pvec_update(pptr_pub, x_curr, pwld_pub);
        if(ROS_GET_SUB_NUM(pub_scan)!=0)
        {
          pcl::PointCloud<PointType> pcl_send;
          pcl_send.reserve(pwld_pub.size());
          for(int i=0; i<pwld_pub.size(); i++)
          {
            Eigen::Vector3d pvec = pwld_pub[i];
            PointType ap;
            ap.x = pvec.x();
            ap.y = pvec.y();
            ap.z = pvec.z();
            pcl_send.push_back(ap);
          }
          pub_pl_func(pcl_send, pub_scan);
        }
        {
          RosPoseStamped pose;
          pose.header.frame_id = "camera_init";
          pose.header.stamp = ROS_TIME_NOW();

          Eigen::Quaterniond q_i2w(x_curr.R);
          pose.pose.position.x = x_curr.p.x();
          pose.pose.position.y = x_curr.p.y();
          pose.pose.position.z = x_curr.p.z();
          pose.pose.orientation.x = q_i2w.x();
          pose.pose.orientation.y = q_i2w.y();
          pose.pose.orientation.z = q_i2w.z();
          pose.pose.orientation.w = q_i2w.w();

          path.poses.push_back(pose);
          path.header.stamp = ROS_TIME_NOW();

          ROS_PUBLISH(pub_trajectory, path);
        }
        t1 = std::chrono::high_resolution_clock::now();
        col_key.push_back("pvec_update");
        col_val.push_back(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()*1e-3);

        t0 = std::chrono::high_resolution_clock::now();
        if(1 == odom_ekf.img_match_status)
        {
          Eigen::Matrix3d rot_cam_to_world = x_curr.R * odom_ekf.rot_cam_to_imu_;
          Eigen::Vector3d pos_cam_in_world = x_curr.R * odom_ekf.pos_cam_in_imu_ + x_curr.p;
          Eigen::Matrix3d rot_world_to_cam = rot_cam_to_world.transpose();
          Eigen::Vector3d pos_world_in_cam = rot_cam_to_world.transpose() * (-pos_cam_in_world);

          
          pcl::PointCloud<PointRGBType> pcl_send;
          if(save_map_duration_>=0. || ROS_GET_SUB_NUM(pub_scan_rgb)!=0)
          {
            pcl_send.reserve(pwld_pub.size());
            for(int i=0; i<pwld_pub.size(); i++)
            {
              Eigen::Vector3d pw = pwld_pub[i];
              PointRGBType ap;
              ap.x = pw.x();
              ap.y = pw.y();
              ap.z = pw.z();
              Eigen::Vector3d pc = rot_world_to_cam * pw + pos_world_in_cam;
              if(pc[2]>0.01)
              {
                Eigen::Vector2d pt = camera_ptr->BearingToPixel(pc);
                if(camera_ptr->IsInBorder(pt, 0, 0))
                {
                  cv::Vec3b bgr = odom_ekf.img_match.at<cv::Vec3b>(cv::Point(pt[0], pt[1]));
                  ap.r = bgr[2];
                  ap.g = bgr[1];
                  ap.b = bgr[0];
                  pcl_send.push_back(ap);
                }
              }
            }
            if(ROS_GET_SUB_NUM(pub_scan_rgb)!=0)
            {
              pub_pl_func(pcl_send, pub_scan_rgb);
            }
          }
          
          if(save_map_duration_>=0.)
          {
            cloud_rgb_save += pcl_send;
            if(map_start_time_<0.)
            {
              map_start_time_ = x_curr.t;
              map_end_time_ = map_start_time_ + 1.e-6;
            }
            else
            {
              map_end_time_ = x_curr.t;
            }
            if(map_end_time_ >= map_start_time_ + save_map_duration_)
            {
              pcl::io::savePCDFileBinary(map_save_path_+ "/cloud_rgb_"+std::to_string(map_start_time_)+"_"+std::to_string(map_end_time_)+".pcd", cloud_rgb_save);
              cloud_rgb_save.clear();
              map_start_time_ = -1.;
              map_end_time_ = -1.;
            }
          }          
        }
        t1 = std::chrono::high_resolution_clock::now();
        col_key.push_back("pub_save_map");
        col_val.push_back(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()*1e-3);

        t0 = std::chrono::high_resolution_clock::now();
        {
          Eigen::Quaterniond q_i2w(x_curr.R);
          Eigen::Vector3d t_i2w(x_curr.p);
          static std::ofstream of_pose(savepath+"/odom_i2w.txt");
          of_pose << std::to_string(x_curr.t) << " " 
                  << t_i2w.x() << " " << t_i2w.y() << " " << t_i2w.z() << " "
                  << q_i2w.x() << " " << q_i2w.y() << " " << q_i2w.z() << " " << q_i2w.w() 
                  << std::endl;  
        }
        t1 = std::chrono::high_resolution_clock::now();
        col_key.push_back("save_pose");
        col_val.push_back(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()*1e-3);

        t0 = std::chrono::high_resolution_clock::now();
        if(1 == odom_ekf.img_match_status)
        {
          Eigen::Matrix3d rot_cam_to_world = x_curr.R * odom_ekf.rot_cam_to_imu_;
          Eigen::Vector3d pos_cam_in_world  = x_curr.R * odom_ekf.pos_cam_in_imu_ + x_curr.p;
          tracking_manager_ptr->AddNewOdomPose(x_curr.t, rot_cam_to_world, pos_cam_in_world);
        }
        t1 = std::chrono::high_resolution_clock::now();
        col_key.push_back("AddNewOdomPose");
        col_val.push_back(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()*1e-3);

        t0 = std::chrono::high_resolution_clock::now();
        win_count++;
        x_buf.push_back(x_curr);
        pvec_buf.push_back(pptr);
        if(win_count > 1)
        {
          imu_pre_buf.push_back(new IMU_PRE(x_buf[win_count-2].bg, x_buf[win_count-2].ba));
          imu_pre_buf[win_count-2]->push_imu(imus);
        }
        t1 = std::chrono::high_resolution_clock::now();
        col_key.push_back("push_imu");
        col_val.push_back(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()*1e-3);
        
        voxhess.clear(); voxhess.win_size = win_size;
        visual_factor.clear(); visual_factor.win_size_ = win_size;

        t0 = std::chrono::high_resolution_clock::now();
        cut_voxel_multi(surf_map, pvec_buf[win_count-1], win_count-1, surf_map_slide, win_size, pwld, sws);
        t1 = std::chrono::high_resolution_clock::now();
        col_key.push_back("cut_voxel_multi");
        col_val.push_back(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()*1e-3);

        t0 = std::chrono::high_resolution_clock::now();
        multi_recut(surf_map_slide, win_count, x_buf, voxhess, sws);
        t1 = std::chrono::high_resolution_clock::now();
        col_key.push_back("multi_recut");
        col_val.push_back(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()*1e-3);

        t0 = std::chrono::high_resolution_clock::now();
        if(1 == odom_ekf.img_match_status)
        {
          Eigen::Matrix3d rot_cam_to_world = x_curr.R * odom_ekf.rot_cam_to_imu_;
          Eigen::Vector3d pos_cam_in_world  = x_curr.R * odom_ekf.pos_cam_in_imu_ + x_curr.p;
          UpdateLandmarkByOdom(surf_map, tracking_ptrs_curr,
            x_curr.t, rot_cam_to_world, pos_cam_in_world, voxel_size, ray_tracking_range[0], ray_tracking_range[1], ray_tracking_all_map, range_img_, range_search_radius_);
          GetVisualFactorByOdom(tracking_ptrs_curr, x_buf, visual_factor);
        }
        t1 = std::chrono::high_resolution_clock::now();
        col_key.push_back("landmark_update");
        col_val.push_back(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()*1e-3);
        
        t0 = std::chrono::high_resolution_clock::now();
        landmark_nums[6] = visual_factor.size();
        landmark_nums[2] = voxhess.size();
        of_landmark << std::to_string(landmark_nums[0]);
        for(int i=1; i<landmark_nums.size(); i++)
        {
          of_landmark << "," << landmark_nums[i];
        }
        of_landmark << std::endl;
        t1 = std::chrono::high_resolution_clock::now();
        col_key.push_back("of_landmark");
        col_val.push_back(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()*1e-3);

        static std::ofstream of_degrade_cnt(savepath + "/degrade_cnt.csv", std::ios::out);
        of_degrade_cnt << std::to_string(x_curr.t) << "," << degrade_cnt << std::endl;
        if(degrade_bound > 0 && degrade_cnt > degrade_bound)
        {
          degrade_cnt = 0;
          system_reset(imus);

          last_pos = x_curr.p; jour = 0;

          reset_flag = 1;

          motion_init_flag = 1;

          continue;
        }
      }

      if(win_count >= win_size)
      {
        t0 = std::chrono::high_resolution_clock::now();
        
        if(localBA_enable)
        {
          if(g_update == 2)
          {
            LI_BA_OptimizerGravity opt_lsv;
            std::vector<double> resis;
            opt_lsv.damping_iter(x_buf, voxhess, imu_pre_buf, resis, &hess, 5);
            printf("g update: %lf %lf %lf: %lf\n", x_buf[0].g[0], x_buf[0].g[1], x_buf[0].g[2], x_buf[0].g.norm());
            g_update = 0;
            x_curr.g = x_buf[win_count-1].g;
          }
          else
          {
            visual_factor.lidar_degrade_ = lidar_degrade_;
            visual_factor.k_no_degrade_ = k_no_degrade_;
            visual_factor.position_disable_thre__ = position_disable_thre__;
            switch(localBA_update_status)
            {
              case 0: // LI-BA
              {
                LI_BA_Optimizer opt_lsv;
                opt_lsv.damping_iter(x_buf, voxhess, imu_pre_buf, &hess, localBA_max_iter_);
              }
              break;
              case 1: // VI-BA
              {
                LVI_BA_Optimizer opt_lsv;
                LidarFactor voxhess_empty(win_size);
                opt_lsv.damping_iter(x_buf, voxhess_empty, visual_factor, imu_pre_buf, &hess, localBA_max_iter_);
              }
              break;
              case 2: // LVI-BA
              {
                std::cout << "voxhess.size():" << voxhess.size()
                  << ", visual_factor.size():" << visual_factor.size()
                  << std::endl;
                LVI_BA_Optimizer opt_lsv;
                opt_lsv.damping_iter(x_buf, voxhess, visual_factor, imu_pre_buf, &hess, localBA_max_iter_);
              }
              break;
              default:
              {
                std::cout << "localBA_update_status:" << localBA_update_status << " is not defined, localBA will not be done." << std::endl;
              }
              break;
            }
          }
        }
        else
        {
          std::cout << "localBA is disable..." << std::endl;
        }

        x_curr.R = x_buf[win_count-1].R;
        x_curr.p = x_buf[win_count-1].p;

        t1 = std::chrono::high_resolution_clock::now();
        col_key.push_back("localBA");
        col_val.push_back(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()*1e-3);

        t0 = std::chrono::high_resolution_clock::now();
        {
          Eigen::Quaterniond q_i2w(x_curr.R);
          Eigen::Vector3d t_i2w(x_curr.p);
          static std::ofstream of_pose(savepath+"/ba_i2w.txt");
          of_pose << std::to_string(x_curr.t) << " " 
                  << t_i2w.x() << " " << t_i2w.y() << " " << t_i2w.z() << " "
                  << q_i2w.x() << " " << q_i2w.y() << " " << q_i2w.z() << " " << q_i2w.w() 
                  << std::endl;  
        }
        t1 = std::chrono::high_resolution_clock::now();
        col_key.push_back("save_pose2");
        col_val.push_back(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()*1e-3);

        t0 = std::chrono::high_resolution_clock::now();
        multi_margi(surf_map_slide, jour, win_count, x_buf, voxhess, sws[0]);
        t1 = std::chrono::high_resolution_clock::now();
        col_key.push_back("multi_margi");
        col_val.push_back(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()*1e-3);

        t0 = std::chrono::high_resolution_clock::now();
        if(!ray_tracking_all_map)
        {
          for(OctoTree* octo:latest_lio_octos)
          {
            octo->latest_lio = false;
          }
        }
        t1 = std::chrono::high_resolution_clock::now();
        col_key.push_back("latest_lio2");
        col_val.push_back(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()*1e-3);
		
        t0 = std::chrono::high_resolution_clock::now();
        if((win_base + win_count) % 10 == 0)
        {
          double spat = (x_curr.p - last_pos).norm();
          if(spat > 0.5)
          {
            jour += spat;
            last_pos = x_curr.p;
            release_flag = true;
          }
        }

        for(int i=0; i<win_size; i++)
        {
          mp[i] += mgsize;
          if(mp[i] >= win_size) mp[i] -= win_size;
        }

        {
          mBuf.lock();
          while(!img_time_buf.empty() && img_time_buf[0]<x_buf[0].t)
          {
            img_time_buf.pop_front();
            img_buf.pop_front();
          }
          tracking_manager_ptr->RemoveOlderMeasurements(x_buf[0].t);
          mBuf.unlock();
        }

        for(int i=mgsize; i<win_count; i++)
        {
          x_buf[i-mgsize] = x_buf[i];
          PVecPtr pvec_tem = pvec_buf[i-mgsize];
          pvec_buf[i-mgsize] = pvec_buf[i];
          pvec_buf[i] = pvec_tem;
        }

        for(int i=win_count-mgsize; i<win_count; i++)
        {
          x_buf.pop_back();
          pvec_buf.pop_back();

          delete imu_pre_buf.front();
          imu_pre_buf.pop_front();
        }

        win_base += mgsize; win_count -= mgsize;
        t1 = std::chrono::high_resolution_clock::now();
        col_key.push_back("localBA_post");
        col_val.push_back(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()*1e-3);
      }
      
      t0 = std::chrono::high_resolution_clock::now();
      if(record_memory_)
      {
        double mem = get_memory();
        std::vector<int> lengths;
        tracking_manager_ptr->GetSize(lengths);
        std::vector<size_t> size_list = getTotalMemory(surf_map);
        std::vector<size_t> size_list_slide = getTotalMemory(surf_map_slide);

        size_t tmp{0};
        for(auto &sw:sws)
        {
          tmp += sw.size();
        }
        if(tmp>0)
        {
          tmp *= sizeof(SlideWindow*) + sizeof(SlideWindow) + sws[0][0]->mem_size();
        }
        memory_log_ << std::to_string(x_curr.t) << "," << mem 
          << "," << lengths[0] << "," << lengths[1] << "," << lengths[2] << "," << lengths[3] 
          << "," << size_list[0] << "," << size_list[1] << "," << double(size_list[2])/(1024.*1024.)
          << "," << size_list_slide[0] << "," << size_list_slide[1] << "," << double(size_list_slide[2])/(1024.*1024.)
          << "," << double(tmp)/(1024.*1024.)
          << std::endl;
      }
      t1 = std::chrono::high_resolution_clock::now();
      col_key.push_back("mem_save");
      col_val.push_back(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()*1e-3);

      end = std::chrono::high_resolution_clock::now();

      if(is_init)
      {
        col_key.push_back("init_total");
        col_val.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()*1e-3);
        if(init_first)
        {
          init_first = false;
          for(std::string key:col_key)
          {
            time_cost_init_ << key << ",";
          }
          time_cost_init_ << std::endl;
        }
        for(double val:col_val)
        {
          time_cost_init_ << val << ",";
        }
        time_cost_init_ << std::endl;
        time_cost_total_ << "init," << col_val.back()
          << "," << std::to_string(std::chrono::duration<double>(start.time_since_epoch()).count())
          << "," << std::to_string(std::chrono::duration<double>(end.time_since_epoch()).count())
          << "," << col_val.front()
          << std::endl;
      }
      else
      {
        col_key.push_back("lvio_total");
        col_val.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()*1e-3);
        if(lvio_first)
        {
          lvio_first = false;
          for(std::string key:col_key)
          {
            time_cost_lvio_ << key << ",";
          }
          time_cost_lvio_ << std::endl;
        }
        for(double val:col_val)
        {
          time_cost_lvio_ << val << ",";
        }
        time_cost_lvio_ << std::endl;
        time_cost_total_ << "lvio," << col_val.back()
          << "," << std::to_string(std::chrono::duration<double>(start.time_since_epoch()).count())
          << "," << std::to_string(std::chrono::duration<double>(end.time_since_epoch()).count())
          << "," << col_val.front()
          << "," << std::to_string(x_curr.t)
          << std::endl;
      }
      
    }

    std::vector<OctoTree *> octos;
    for(auto iter=surf_map.begin(); iter!=surf_map.end(); iter++)
    {
      iter->second->tras_ptr(octos);
      iter->second->clear_slwd(sws[0]);
      delete iter->second;
    }

    for(int i=0; i<octos.size(); i++)
      delete octos[i];
    octos.clear();

    for(int i=0; i<sws[0].size(); i++)
      delete sws[0][i];
    sws[0].clear();
    malloc_trim(0);
  }

};

int main(int argc, char **argv)
{
  if(argc!=2)
  {
    std::cout << "Usage: ./voxelslam <config_folder_name>" << std::endl;
    return -1;
  }

  const std::string cfg_path = std::string(PROJECT_PATH) + "/config/" + argv[1];
  const std::string cfg_odom = cfg_path + "/odom.yaml";
  std::cout << "cfg_path:" << cfg_path << std::endl;

  YAML::Node odom_node = YAML::LoadFile(cfg_odom);
  string lid_topic, imu_topic, img_topic, compressed_img_topic;
  YamlRead(odom_node["General"], "lid_topic", lid_topic);
  YamlRead(odom_node["General"], "imu_topic", imu_topic);
  YamlRead(odom_node["General"], "img_topic", img_topic);
  YamlRead(odom_node["General"], "compressed_img_topic", compressed_img_topic);
  std::cout << "lidar_topic:" << lid_topic << ", imu_topic:" << imu_topic
    << ", img_topic:" << img_topic << ", compressed_img_topic:" << compressed_img_topic
    << std::endl;
  if(imu_topic.empty())
  {
    std::cout << "imu_topic is empty!!!" << std::endl;
    exit(1);
  }
  if(lid_topic.empty())
  {
    std::cout << "lid_topic is empty!!!" << std::endl;
    exit(1);
  }

  VOXEL_SLAM vs(cfg_path);
  mp = new int[vs.win_size];
  for(int i=0; i<vs.win_size; i++)
    mp[i] = i;
  
  std::thread odom_thread(&VOXEL_SLAM::thd_odometry_localmapping, &vs);
  
  ROS_INIT(argc, argv, "cmn_voxel");
  RosNode ros_node = ROS_NODE("cmn_voxel");

  sub_imu = ROS_SUBSCRIBE(ros_node, RosImu, imu_topic, 80000, imu_handler);
  sub_pcl = ROS_SUBSCRIBE(ros_node, RosCloud, lid_topic, 1000, pcl_handler); 
  if(!img_topic.empty())
  {
    sub_cam = ROS_SUBSCRIBE(ros_node, RosImage, img_topic, 10, monocular_handler);
  }
  else if(!compressed_img_topic.empty())
  {
    sub_cam_compressed = ROS_SUBSCRIBE(ros_node, RosCompressedImage, compressed_img_topic, 10, monocular_compressed_handler);
  }
  pub_cmap = ROS_ADVERTISE(ros_node, RosCloud, "/map_cmap", 100);
  pub_pmap = ROS_ADVERTISE(ros_node, RosCloud, "/map_pmap", 100);
  pub_scan = ROS_ADVERTISE(ros_node, RosCloud, "/map_scan", 100);
  pub_init = ROS_ADVERTISE(ros_node, RosCloud, "/map_init", 100);
  pub_curr_path = ROS_ADVERTISE(ros_node, RosCloud, "/map_path", 100);
  pub_trajectory = ROS_ADVERTISE(ros_node, RosPath, "/trajectory", 10);
  pub_prev_path = ROS_ADVERTISE(ros_node, RosCloud, "/map_true", 100);

  pub_scan_rgb = ROS_ADVERTISE(ros_node, RosCloud, "/map_scan_rgb", 100);

  path.header.frame_id = "camera_init";

  ROS_SPIN(ros_node);
  if(odom_thread.joinable())
  {
    vs.odom_alive_ = false;
    odom_thread.join();
  }
  ROS_SHUTDOWN();
  std::cout << "voxelslam exit" << std::endl;
  return 0;
}

