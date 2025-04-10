#include "DenseMapping.h"

namespace ORB_SLAM3
{

void simple_ground_filter(const PCLPointCloud& pc, PCLPointCloud& ground, PCLPointCloud& nonground){
        pcl::PassThrough<pcl::PointXYZ> simple_filter;
        simple_filter.setFilterFieldName("y");
        simple_filter.setFilterLimits(-0.1, 100);
        simple_filter.setInputCloud(pc.makeShared());
        simple_filter.filter(ground);
        simple_filter.setNegative(true);
        simple_filter.filter(nonground);
}

octomap::point3d DenseMapping::getKFGlobalPC(KeyFrame* pKF, PCLPointCloud& ground, PCLPointCloud& nonground){
    ground.clear();
    nonground.clear();
    pKF->SetNotErase();
    auto tmp = pKF->GetPointCloud();
    // 转换到世界坐标下==== (y==down, x==right, z==forward)
    Eigen::Isometry3d T = ORB_SLAM3::Converter::toSE3Quat(pKF->GetPoseInverse());
    PCLPointCloud pc;
    float agent_height = settings->camHeight(); //must use the correct robot height
    //for easier processing ground, shift agent height to -height so ground is at y==0 (point down)
    T(1,3) -= agent_height;
    pcl::transformPointCloud(*tmp, pc, T.matrix());
    octomap::point3d sensorOrigin = octomap::point3d(T(0,3), T(1,3), T(2,3));
    //auto Tw = T.inverse();
    //octomap::point3d sensorOrigin = octomap::point3d(Tw(0,3), Tw(1,3), Tw(2,3));
    pKF->SetErase();

    //may filter pc based on x,y,z bounding box and remove NaNs 
    pcl::PassThrough<pcl::PointXYZ> simple_filter;
    simple_filter.setFilterFieldName("y");
    simple_filter.setFilterLimits(-agent_height-0.1, 0.1);
    simple_filter.setInputCloud(pc.makeShared());
    simple_filter.filter(pc);
    //using filter pcl::PassThrough<pcl::PointXYZ> pass;

    // split ground and nonground points
    if(settings->simpleGround())
    {
        // 这里可以简单剔除掉 y轴方向 > 机器人高度的 点，加速去除平面=======
        //take ground and nonground points from pointcloud based on it's height
        simple_ground_filter(pc, ground, nonground);
    }else{
        FilterGroundPlane(pc, ground, nonground);
    }
    if(settings->map2D()){
    }
    return sensorOrigin;
}

void DenseMapping::Run(){
    KeyFrame* pKF = NULL;
    PCLPointCloud ground, nonground;
    while(1){
        if(isCloseLoop()){
            std::cout<< " closing loop, rebuild octree" << std::endl;
            //todo: add timing of this block  
            {
                unique_lock<mutex> lock(mMutexOctree);
                m_octree->clear();
                m_gridmap.clear();
            }
            vector<KeyFrame*> vKFs = mpAtlas->GetCurrentMap()->GetAllKeyFrames();
            {
                unique_lock<mutex> lock(mMutexQueue);
                mKeyFrameQueue.clear();
                int N = vKFs.size();
                //todo: don't clear and rebuild the updated frames only!!!
                int step = N/30;//max downsampled to 30 frames total
                for(int i=0; i<N; i+=step){
                    mKeyFrameQueue.push_back(vKFs[i]);
                }
            }
            LoopClosed();
        }

        if(CheckNewKeyFrame()){
            {
                unique_lock<mutex> lock(mMutexQueue);
                pKF = mKeyFrameQueue.front();
                mKeyFrameQueue.pop_front();
            }
            auto origin = getKFGlobalPC(pKF, ground, nonground);
            //lock updating m_octree
            unique_lock<mutex> lock(mMutexOctree);
            //m_octree->insertPointCloud(nonground, origin);
            InsertScan(origin, ground, nonground);
            ground.clear();
            nonground.clear();

            build2DMap();
        }
        else //todo, add if isNewMap octree.clear
            usleep(2000);
    }
}

void DenseMapping::InsertScan(const octomap::point3d& sensorOrigin, 
    const PCLPointCloud &ground, const PCLPointCloud &nonground)
{
    // 坐标转换到 key
    if(!m_octree->coordToKeyChecked(sensorOrigin, m_updateBBXMin)||
        !m_octree->coordToKeyChecked(sensorOrigin, m_updateBBXMax))
     {
            printf("coulde not generate key for origin\n");
     }

     octomap::KeySet free_cells, occupied_cells;

    // insert ground points only as free:
    for (auto it = ground.begin(); it != ground.end(); ++it) {
        octomap::point3d point(it->x, it->y, it->z);
        // maxrange check
        if ((m_maxRange > 0.0) && ((point - sensorOrigin).norm() > m_maxRange) ) {
            point = sensorOrigin + (point - sensorOrigin).normalized() * m_maxRange;
        }

        // only clear space (ground points)
        if (m_octree->computeRayKeys(sensorOrigin, point, m_keyRay)) {
            free_cells.insert(m_keyRay.begin(), m_keyRay.end());
        } 

        octomap::OcTreeKey endKey;
        if (m_octree->coordToKeyChecked(point, endKey)) {
            updateMinKey(endKey, m_updateBBXMin);
            updateMaxKey(endKey, m_updateBBXMax);
        } else {
              printf("could not generator key for endpoint");
        }
    } 

    // all other points: free on ray, occupied on endpoint:
    for (auto it = nonground.begin(); it != nonground.end(); ++it) {
        octomap::point3d point(it->x, it->y, it->z);
        // maxrange check            
        if ((m_maxRange < 0.0) || ((point - sensorOrigin).norm() <= m_maxRange)) {
            // free cells
            if (m_octree->computeRayKeys(sensorOrigin, point, m_keyRay)) {
                free_cells.insert(m_keyRay.begin(), m_keyRay.end());
            }
            // occupied endpoint
            octomap::OcTreeKey key;
            if (m_octree->coordToKeyChecked(point, key)) {
                occupied_cells.insert(key);
                updateMinKey(key, m_updateBBXMin);
                updateMaxKey(key, m_updateBBXMax); 
            }
        } else {
            // ray longer than maxrange:;  
            octomap::point3d new_end = sensorOrigin +
                (point - sensorOrigin).normalized() * m_maxRange;
            if (m_octree->computeRayKeys(sensorOrigin, new_end, m_keyRay)) {
                free_cells.insert(m_keyRay.begin(), m_keyRay.end());
                octomap::OcTreeKey endKey;
                if (m_octree->coordToKeyChecked(new_end, endKey)) {
                    free_cells.insert(endKey);
                    updateMinKey(endKey, m_updateBBXMin);
                    updateMaxKey(endKey, m_updateBBXMax);
                } else {
                    printf("could not generator key for endpoint");
                }
            }
        }
    }
    // mark free cells only if not seen occupied in this cloud
    for(auto it = free_cells.begin(), end=free_cells.end();
        it!= end; ++it){
        if (occupied_cells.find(*it) == occupied_cells.end()){
            m_octree->updateNode(*it, false);
        }
    }

    // now mark all occupied cells:
    for (auto it = occupied_cells.begin(),
                end=occupied_cells.end(); it!= end; it++) {
        m_octree->updateNode(*it, true);
    }
     m_octree->prune();
     m_treeDepth = m_octree->getTreeDepth();
     //printf("tree depth = %d\n", m_treeDepth); //==16
}

void DenseMapping::FilterGroundPlane(const PCLPointCloud& pc, 
    PCLPointCloud& ground, PCLPointCloud& nonground) const{
        ground.header = pc.header;
        nonground.header = pc.header;

        float m_groundFilterDistance = 0.05;
        float m_groundFilterAngle = 0.15;
        float m_groundFilterPlaneDistance = 0.07;

        if (pc.size() < 50){
            printf("Pointcloud is too small, skipping ground plane extraction");
            simple_ground_filter(pc, ground, nonground);
        } else {
            // plane detection for ground plane removal:
            pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
            pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

            // Create the segmentation object and set up:
            pcl::SACSegmentation<pcl::PointXYZ> seg;
            seg.setOptimizeCoefficients (true);
            // TODO:
            // maybe a filtering based on the surface normals might be more robust / accurate?
            seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
            seg.setMethodType(pcl::SAC_RANSAC);
            seg.setMaxIterations(100);
            seg.setDistanceThreshold (m_groundFilterDistance);
            //seg.setAxis(Eigen::Vector3f(0, 0, 1));
            seg.setAxis(Eigen::Vector3f(0, -1, 0));
            seg.setEpsAngle(m_groundFilterAngle);

            PCLPointCloud cloud_filtered(pc);
            // Create the filtering object
            pcl::ExtractIndices<pcl::PointXYZ> extract;
            bool groundPlaneFound = false;

            while (cloud_filtered.size() > 10 && !groundPlaneFound) {
                seg.setInputCloud(cloud_filtered.makeShared());
                seg.segment (*inliers, *coefficients);
                if (inliers->indices.size () == 0) {
                    //printf("PCL segmentation did not find any plane.\n");
                    break;
                }

                extract.setInputCloud(cloud_filtered.makeShared());
                extract.setIndices(inliers);

                if (std::abs(coefficients->values.at(3)) < m_groundFilterPlaneDistance) {
                        // printf("Ground plane found: %zu/%zu inliers. Coeff: %f %f %f %f \n",
                        // inliers->indices.size(), cloud_filtered.size(),
                        // coefficients->values.at(0), coefficients->values.at(1),
                        // coefficients->values.at(2), coefficients->values.at(3));
                    extract.setNegative(false);
                    extract.filter(ground);

                    // remove ground points from full pointcloud:
                    // workaround for PCL bug:
                    if(inliers->indices.size() != cloud_filtered.size()) {
                        extract.setNegative(true);
                        //PCLPointCloud cloud_out; //bug when doing assignement
                        //extract.filter(cloud_out);
                        //nonground += cloud_out;
                        //cloud_filtered = cloud_out;
                        cloud_filtered.points.clear();
                        extract.filter(cloud_filtered);
                        nonground += cloud_filtered;
                    }
                    groundPlaneFound = true;
                } else {
                    // printf("Horizontal plane (not ground) found: %zu/%zu inliers. Coeff: %f %f %f %f\n",
                    //             inliers->indices.size(), cloud_filtered.size(),
                    //           coefficients->values.at(0), coefficients->values.at(1),
                    //             coefficients->values.at(2), coefficients->values.at(3));
                    extract.setNegative (false);
                    //PCLPointCloud cloud_out;
                    //extract.filter(cloud_out);
                    //nonground +=cloud_out;
                    extract.filter(nonground);

                    // remove current plane from scan for next iteration:
                    // workaround for PCL bug:
                    if(inliers->indices.size() != cloud_filtered.size()){
                        extract.setNegative(true);
                        cloud_filtered.points.clear();
                        extract.filter(cloud_filtered);
                    }
                    else
                    {
                        cloud_filtered.points.clear();
                    }
                }
            }
            // TODO: also do this if overall starting pointcloud too small?
            if (!groundPlaneFound)
            { // no plane found or remaining points too small
                // printf("No ground plane found in scan. \n");
                // do a rough fitlering on height to prevent spurious obstacles
                simple_ground_filter(pc, ground, nonground);
            }
        }
}

void DenseMapping::build2DMap() {
    double minX, minY, minZ;
    double maxX, maxY, maxZ;
    m_octree->getMetricMin(minX, minY, minZ);
    m_octree->getMetricMax(maxX, maxY, maxZ);
    m_gridmap.check_update(minZ, minX, maxZ, maxX);

    octomap::point3d minPt(m_gridmap.m_miny, minY, m_gridmap.m_minx);
    auto minKey = m_octree->coordToKey(minPt, m_treeDepth);

    //octomap has (x==right, y==down, z==forward)!!!!
    //map to z -> x, -x -> y (x->y remember to flip by the user), (x forward, y left, z up)

    for (auto it=m_octree->begin(m_treeDepth); it!=m_octree->end(); ++it) {
        bool inUpdatedBBX = isInUpdateBBX(it);
        if(!inUpdatedBBX)continue;
        update2DMap(it, m_octree->isNodeOccupied(*it), minKey);
    }
}

void DenseMapping::update2DMap(const octomap::OcTree::iterator& it, bool occupied, octomap::OcTreeKey minKey) {
    if (it.getDepth() == m_treeDepth){
        auto key = it.getKey();
        auto idx = m_gridmap.idx(key[2]-minKey[2], 
            key[0]-minKey[0]);
        if (occupied) {
            m_gridmap.data[idx] = 1;
        } else if (m_gridmap.data[idx] == -1){
            m_gridmap.data[idx] = 0;
        }
    } else {
        int intSize = 1 << (m_treeDepth - it.getDepth());
        octomap::OcTreeKey minKey=it.getIndexKey();
        for(int dx = 0; dx < intSize; dx++) {
            int i =  dx;
            for(int dy = 0; dy < intSize; dy++){
                auto idx = m_gridmap.idx(i, dy);
                if (occupied) {
                    m_gridmap.data[idx] = 1;
                } else if (m_gridmap.data[idx] == -1) {
                    m_gridmap.data[idx] = 0;
                }
            }
        }
    }
}   


}
