#ifndef DENSEMAPPING_H
#define DENSEMAPPING_H

#include "KeyFrame.h"
#include "Atlas.h"
#include "Settings.h"
#include <mutex>

// octomap
#include <octomap/ColorOcTree.h>
#include <octomap/Pointcloud.h>
#include <octomap/octomap.h>
// pcl
#include <pcl/io/pcd_io.h>// 读写
#include <pcl/common/transforms.h>// 点云坐标变换
#include <pcl/point_types.h>      // 点类型
#include <pcl/filters/voxel_grid.h>// 体素格滤波
#include <pcl/filters/passthrough.h>//  直通滤波
#include <pcl/sample_consensus/method_types.h>// 采样一致性，采样方法
#include <pcl/sample_consensus/model_types.h>// 模型
#include <pcl/segmentation/sac_segmentation.h>// 采样一致性分割
#include <pcl/filters/extract_indices.h>// 提取点晕索引

using PCLPointCloud = pcl::PointCloud<pcl::PointXYZ>;

namespace ORB_SLAM3
{

class System;


//2D occupany grid map
struct GridMap{
    GridMap():GridMap(200, 200, 0.5){}
    GridMap(int width, int height, double res):data(width* height, -1){
        m_width = width;
        m_height = height;
        m_res = res;
        m_minx = -m_width/2.0 * m_res;
        m_miny = -m_height/2.0 * m_res;
    }

    void check_update(double x_min, double y_min, double x_max, double y_max){
        //map grows out of bondary, then needs to create a new map and update
        if(x_min<=m_minx || x_max>=m_minx+m_width*m_res || y_min<=m_miny || y_max>=m_miny+m_height*m_res){
            std::cout<<"expaneded occ map!"<<std::endl;
            int II = int(D/m_res);
            if(x_min<=m_minx){
                int old_width = m_width;
                m_minx = x_min - II*m_res;
                m_width += II;
                //make a deep copy of data
                std::vector<short> copy(data);
                data.resize(m_width * m_height);
                data.assign(data.size(), -1);
                for(int j=0; j < m_height; j++){
                    for (int i = II; i < m_width; i++){
                        data[idx(i, j)] = copy[j*old_width + i -II];
                    }
                }
            }
            if(x_max>=m_minx+m_width*m_res){
                int old_width = m_width;
                m_width += II;
                std::vector<short> copy(data);
                data.resize(m_width * m_height);
                data.assign(data.size(), -1);
                for(int j=0; j < m_height; j++){
                    for (int i = 0; i < m_width-II; i++){
                        data[idx(i, j)] = copy[j*old_width+i];
                    }
                }
            }
            if(y_min<=m_miny){
                m_miny = y_min - II*m_res;
                m_height += II;
                std::vector<short> copy(data);
                data.resize(m_width * m_height);
                data.assign(data.size(), -1);
                for(int j=II; j < m_height; j++){
                    for (int i = 0; i < m_width; i++){
                        data[idx(i, j)] = copy[idx(i, j-II)];
                    }
                }
            }
            if(y_max>=m_miny+m_height*m_res){
                m_height += II;
                std::vector<short> copy(data);
                data.resize(m_width * m_height);
                data.assign(data.size(), -1);
                for(int j=0; j < m_height-II; j++){
                    for (int i = 0; i < m_width; i++){
                        data[idx(i, j)] = copy[idx(i, j)];
                    }
                }
            }
        }
    }
    void clear(){
        data.assign(data.size(), -1);
    }

    inline unsigned idx(int i, int j) const{
        return m_width * j + i;
    }
    int m_width, m_height;
    double m_res;//resolution
    double m_minx, m_miny;
    std::vector<short> data;
    const double D = 5.0;
};

class DenseMapping
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    DenseMapping(System* pSys, Atlas* pAtlas, Settings* settings_):
        mpSys(pSys),
        mpAtlas(pAtlas),
        settings(settings_),
        m_octree(new octomap::OcTree(settings->gridSize())),
        m_gridmap(400, 400, settings->gridSize()){
            m_octree->setProbHit(0.7);
            m_octree->setProbMiss(0.4);
            m_octree->setClampingThresMax(0.95);
            m_octree->setClampingThresMin(0.05);
            m_treeDepth = m_octree->getTreeDepth();
            prevKFpose = Sophus::SE3f::transX(10000.0);
            m_maxRange = settings->maxRange();
    }
    void Run();
    void InsertKeyFrame(KeyFrame* pKF){
        //avoid inserting too many keyframes so can run faster
        auto currPose = pKF->GetPoseInverse();
        //difference between currPose and prevKFpose
        auto diff = currPose.inverse() * prevKFpose;
        double t = pKF->mTimeStamp;
        double dt = t - mTimeStamp;
        if(dt<0.01)return;
        if (diff.translation().norm() < 0.40 && diff.rotationMatrix().eulerAngles(0, 1, 2).norm() < 3.14/8 && dt<0.1)return;
        {
            unique_lock<mutex> lock(mMutexQueue);
            if(pKF->mnId!=0)
                mKeyFrameQueue.push_back(pKF);
        }
        prevKFpose = currPose;
        mTimeStamp = t;
    }

    bool CheckNewKeyFrame(){
        unique_lock<mutex> lock(mMutexQueue);
        if (mKeyFrameQueue.size()>3){
            printf("Dense Mapping lag behind!!, queue size = %d\n", mKeyFrameQueue.size());
        }
        return !mKeyFrameQueue.empty();
    }

    void EmptyQueue(){
        unique_lock<mutex> lock(mMutexQueue);
        mKeyFrameQueue.clear();
    }

    void RequestCloseLoop(){
        unique_lock<mutex> lock(mMutexCloseLoop);
        mbCloseLoop = true;
    }

    void LoopClosed(){
        unique_lock<mutex> lock(mMutexCloseLoop);
        mbCloseLoop = false;
    }

    bool isCloseLoop(){
        unique_lock<mutex> lock(mMutexCloseLoop);
        return mbCloseLoop;
    }

    ~DenseMapping(){
        delete m_octree;
    }

    auto getOctreeCopy() { 
        //return a copy of the octree
        unique_lock<mutex> lock(mMutexOctree);
        return octomap::OcTree(*m_octree);
    }

    octomap::point3d getKFGlobalPC(KeyFrame *pKF, PCLPointCloud &ground, PCLPointCloud &nonground);
    void InsertScan(const octomap::point3d &sensorOrigin, const PCLPointCloud &ground, const PCLPointCloud &nonground);
    void FilterGroundPlane( const PCLPointCloud& pc, PCLPointCloud& ground, PCLPointCloud& nonground) const;
    void update2DMap(const octomap::OcTree::iterator& it, bool occupied, octomap::OcTreeKey minKey);
    void build2DMap();
    /// Test if key is within update area of map (2D, ignores height)
    inline bool isInUpdateBBX(const octomap::OcTree::iterator &it) const
    {
        // 2^(tree_depth-depth) voxels wide:
        unsigned voxelWidth = (1 << (m_treeDepth - it.getDepth()));
        octomap::OcTreeKey key = it.getIndexKey(); // lower corner of voxel
        return (key[0] + voxelWidth >= m_updateBBXMin[0] && key[1] + voxelWidth >= m_updateBBXMin[1] && key[0] <= m_updateBBXMax[0] && key[1] <= m_updateBBXMax[1]);
    }

    inline static void updateMinKey(const octomap::OcTreeKey &in,
                                    octomap::OcTreeKey &min)
    {
        for (unsigned i = 0; i < 3; ++i)
        {
            min[i] = std::min(in[i], min[i]);
        }
    }

    inline static void updateMaxKey(const octomap::OcTreeKey &in,
                                    octomap::OcTreeKey &max)
    {
        for (unsigned i = 0; i < 3; ++i)
        {
            max[i] = std::max(in[i], max[i]);
        }
    }

    GridMap& get2DOccMap() { return m_gridmap;}

private:
    System* mpSys;
    Atlas* mpAtlas;
    Settings* settings;
    octomap::OcTree *m_octree;
    GridMap m_gridmap;
    bool mbCloseLoop = false;
    std::mutex mMutexQueue, mMutexCloseLoop;
    std::list<KeyFrame*> mKeyFrameQueue;
    std::mutex mMutexOctree;
    Sophus::SE3f prevKFpose;
    double mTimeStamp = 0;

    octomap::KeyRay m_keyRay;  // temp storage for ray casting
    octomap::OcTreeKey m_updateBBXMin;
    octomap::OcTreeKey m_updateBBXMax;
    unsigned m_treeDepth;
    double m_maxRange;
};

}

#endif