/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include "MapDrawer.h"
#include "MapPoint.h"
#include "KeyFrame.h"
//#include <pangolin/pangolin.h>
#include <mutex>

namespace ORB_SLAM3
{


MapDrawer::MapDrawer(Atlas* pAtlas, const string &strSettingPath, Settings* settings):mpAtlas(pAtlas)
{
    if(settings){
        newParameterLoader(settings);
    }
    else{
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
        bool is_correct = ParseViewerParamFile(fSettings);

        if(!is_correct)
        {
            std::cerr << "**ERROR in the config file, the format is not correct**" << std::endl;
            try
            {
                throw -1;
            }
            catch(exception &e)
            {

            }
        }
    }
}

void MapDrawer::newParameterLoader(Settings *settings) {
    mKeyFrameSize = settings->keyFrameSize();
    mKeyFrameLineWidth = settings->keyFrameLineWidth();
    mGraphLineWidth = settings->graphLineWidth();
    mPointSize = settings->pointSize();
    mCameraSize = settings->cameraSize();
    mCameraLineWidth  = settings->cameraLineWidth();
}

bool MapDrawer::ParseViewerParamFile(cv::FileStorage &fSettings)
{
    bool b_miss_params = false;

    cv::FileNode node = fSettings["Viewer.KeyFrameSize"];
    if(!node.empty())
    {
        mKeyFrameSize = node.real();
    }
    else
    {
        std::cerr << "*Viewer.KeyFrameSize parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.KeyFrameLineWidth"];
    if(!node.empty())
    {
        mKeyFrameLineWidth = node.real();
    }
    else
    {
        std::cerr << "*Viewer.KeyFrameLineWidth parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.GraphLineWidth"];
    if(!node.empty())
    {
        mGraphLineWidth = node.real();
    }
    else
    {
        std::cerr << "*Viewer.GraphLineWidth parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.PointSize"];
    if(!node.empty())
    {
        mPointSize = node.real();
    }
    else
    {
        std::cerr << "*Viewer.PointSize parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.CameraSize"];
    if(!node.empty())
    {
        mCameraSize = node.real();
    }
    else
    {
        std::cerr << "*Viewer.CameraSize parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.CameraLineWidth"];
    if(!node.empty())
    {
        mCameraLineWidth = node.real();
    }
    else
    {
        std::cerr << "*Viewer.CameraLineWidth parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    return !b_miss_params;
}

void MapDrawer::DrawMapPoints()
{
    Map* pActiveMap = mpAtlas->GetCurrentMap();
    if(!pActiveMap)
        return;

    const vector<MapPoint*> &vpMPs = pActiveMap->GetAllMapPoints();
    const vector<MapPoint*> &vpRefMPs = pActiveMap->GetReferenceMapPoints();

    set<MapPoint*> spRefMPs(vpRefMPs.begin(), vpRefMPs.end());

    if(vpMPs.empty())
        return;

    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(0.0,0.0,0.0);

    for(size_t i=0, iend=vpMPs.size(); i<iend;i++)
    {
        if(vpMPs[i]->isBad() || spRefMPs.count(vpMPs[i]))
            continue;
        Eigen::Matrix<float,3,1> pos = vpMPs[i]->GetWorldPos();
        glVertex3f(pos(0),pos(1),pos(2));
    }
    glEnd();

    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(1.0,0.0,0.0);

    for(set<MapPoint*>::iterator sit=spRefMPs.begin(), send=spRefMPs.end(); sit!=send; sit++)
    {
        if((*sit)->isBad())
            continue;
        Eigen::Matrix<float,3,1> pos = (*sit)->GetWorldPos();
        glVertex3f(pos(0),pos(1),pos(2));

    }

    glEnd();
}

void MapDrawer::DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph, const bool bDrawInertialGraph, const bool bDrawOptLba)
{
    const float &w = mKeyFrameSize;
    const float h = w*0.75;
    const float z = w*0.6;

    Map* pActiveMap = mpAtlas->GetCurrentMap();
    // DEBUG LBA
    std::set<long unsigned int> sOptKFs = pActiveMap->msOptKFs;
    std::set<long unsigned int> sFixedKFs = pActiveMap->msFixedKFs;

    if(!pActiveMap)
        return;

    const vector<KeyFrame*> vpKFs = pActiveMap->GetAllKeyFrames();

    if(bDrawKF)
    {
        for(size_t i=0; i<vpKFs.size(); i++)
        {
            KeyFrame* pKF = vpKFs[i];
            Eigen::Matrix4f Twc = pKF->GetPoseInverse().matrix();
            unsigned int index_color = pKF->mnOriginMapId;

            glPushMatrix();

            glMultMatrixf((GLfloat*)Twc.data());

            if(!pKF->GetParent()) // It is the first KF in the map
            {
                glLineWidth(mKeyFrameLineWidth*5);
                glColor3f(1.0f,0.0f,0.0f);
                glBegin(GL_LINES);
            }
            else
            {
                //cout << "Child KF: " << vpKFs[i]->mnId << endl;
                glLineWidth(mKeyFrameLineWidth);
                if (bDrawOptLba) {
                    if(sOptKFs.find(pKF->mnId) != sOptKFs.end())
                    {
                        glColor3f(0.0f,1.0f,0.0f); // Green -> Opt KFs
                    }
                    else if(sFixedKFs.find(pKF->mnId) != sFixedKFs.end())
                    {
                        glColor3f(1.0f,0.0f,0.0f); // Red -> Fixed KFs
                    }
                    else
                    {
                        glColor3f(0.0f,0.0f,1.0f); // Basic color
                    }
                }
                else
                {
                    glColor3f(0.0f,0.0f,1.0f); // Basic color
                }
                glBegin(GL_LINES);
            }

            glVertex3f(0,0,0);
            glVertex3f(w,h,z);
            glVertex3f(0,0,0);
            glVertex3f(w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,h,z);

            glVertex3f(w,h,z);
            glVertex3f(w,-h,z);

            glVertex3f(-w,h,z);
            glVertex3f(-w,-h,z);

            glVertex3f(-w,h,z);
            glVertex3f(w,h,z);

            glVertex3f(-w,-h,z);
            glVertex3f(w,-h,z);
            glEnd();

            glPopMatrix();

            glEnd();
        }
    }

    if(bDrawGraph)
    {
        glLineWidth(mGraphLineWidth);
        glColor4f(0.0f,1.0f,0.0f,0.6f);
        glBegin(GL_LINES);

        // cout << "-----------------Draw graph-----------------" << endl;
        for(size_t i=0; i<vpKFs.size(); i++)
        {
            // Covisibility Graph
            const vector<KeyFrame*> vCovKFs = vpKFs[i]->GetCovisiblesByWeight(100);
            Eigen::Vector3f Ow = vpKFs[i]->GetCameraCenter();
            if(!vCovKFs.empty())
            {
                for(vector<KeyFrame*>::const_iterator vit=vCovKFs.begin(), vend=vCovKFs.end(); vit!=vend; vit++)
                {
                    if((*vit)->mnId<vpKFs[i]->mnId)
                        continue;
                    Eigen::Vector3f Ow2 = (*vit)->GetCameraCenter();
                    glVertex3f(Ow(0),Ow(1),Ow(2));
                    glVertex3f(Ow2(0),Ow2(1),Ow2(2));
                }
            }

            // Spanning tree
            KeyFrame* pParent = vpKFs[i]->GetParent();
            if(pParent)
            {
                Eigen::Vector3f Owp = pParent->GetCameraCenter();
                glVertex3f(Ow(0),Ow(1),Ow(2));
                glVertex3f(Owp(0),Owp(1),Owp(2));
            }

            // Loops
            set<KeyFrame*> sLoopKFs = vpKFs[i]->GetLoopEdges();
            for(set<KeyFrame*>::iterator sit=sLoopKFs.begin(), send=sLoopKFs.end(); sit!=send; sit++)
            {
                if((*sit)->mnId<vpKFs[i]->mnId)
                    continue;
                Eigen::Vector3f Owl = (*sit)->GetCameraCenter();
                glVertex3f(Ow(0),Ow(1),Ow(2));
                glVertex3f(Owl(0),Owl(1),Owl(2));
            }
        }

        glEnd();
    }

    if(bDrawInertialGraph && pActiveMap->isImuInitialized())
    {
        glLineWidth(mGraphLineWidth);
        glColor4f(1.0f,0.0f,0.0f,0.6f);
        glBegin(GL_LINES);

        //Draw inertial links
        for(size_t i=0; i<vpKFs.size(); i++)
        {
            KeyFrame* pKFi = vpKFs[i];
            Eigen::Vector3f Ow = pKFi->GetCameraCenter();
            KeyFrame* pNext = pKFi->mNextKF;
            if(pNext)
            {
                Eigen::Vector3f Owp = pNext->GetCameraCenter();
                glVertex3f(Ow(0),Ow(1),Ow(2));
                glVertex3f(Owp(0),Owp(1),Owp(2));
            }
        }

        glEnd();
    }

    vector<Map*> vpMaps = mpAtlas->GetAllMaps();

    if(bDrawKF)
    {
        for(Map* pMap : vpMaps)
        {
            if(pMap == pActiveMap)
                continue;

            vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();

            for(size_t i=0; i<vpKFs.size(); i++)
            {
                KeyFrame* pKF = vpKFs[i];
                Eigen::Matrix4f Twc = pKF->GetPoseInverse().matrix();
                unsigned int index_color = pKF->mnOriginMapId;

                glPushMatrix();

                glMultMatrixf((GLfloat*)Twc.data());

                if(!vpKFs[i]->GetParent()) // It is the first KF in the map
                {
                    glLineWidth(mKeyFrameLineWidth*5);
                    glColor3f(1.0f,0.0f,0.0f);
                    glBegin(GL_LINES);
                }
                else
                {
                    glLineWidth(mKeyFrameLineWidth);
                    glColor3f(mfFrameColors[index_color][0],mfFrameColors[index_color][1],mfFrameColors[index_color][2]);
                    glBegin(GL_LINES);
                }

                glVertex3f(0,0,0);
                glVertex3f(w,h,z);
                glVertex3f(0,0,0);
                glVertex3f(w,-h,z);
                glVertex3f(0,0,0);
                glVertex3f(-w,-h,z);
                glVertex3f(0,0,0);
                glVertex3f(-w,h,z);

                glVertex3f(w,h,z);
                glVertex3f(w,-h,z);

                glVertex3f(-w,h,z);
                glVertex3f(-w,-h,z);

                glVertex3f(-w,h,z);
                glVertex3f(w,h,z);

                glVertex3f(-w,-h,z);
                glVertex3f(w,-h,z);
                glEnd();

                glPopMatrix();
            }
        }
    }
}

void MapDrawer::DrawCurrentCamera(pangolin::OpenGlMatrix &Twc)
{
    const float &w = mCameraSize;
    const float h = w*0.75;
    const float z = w*0.6;

    glPushMatrix();

#ifdef HAVE_GLES
        glMultMatrixf(Twc.m);
#else
        glMultMatrixd(Twc.m);
#endif

    glLineWidth(mCameraLineWidth);
    glColor3f(0.0f,1.0f,0.0f);
    glBegin(GL_LINES);
    glVertex3f(0,0,0);
    glVertex3f(w,h,z);
    glVertex3f(0,0,0);
    glVertex3f(w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,h,z);

    glVertex3f(w,h,z);
    glVertex3f(w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(-w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(w,h,z);

    glVertex3f(-w,-h,z);
    glVertex3f(w,-h,z);
    glEnd();

    glPopMatrix();
}


void MapDrawer::SetCurrentCameraPose(const Sophus::SE3f &Tcw)
{
    unique_lock<mutex> lock(mMutexCamera);
    mCameraPose = Tcw.inverse();
}

void MapDrawer::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M, pangolin::OpenGlMatrix &MOw)
{
    Eigen::Matrix4f Twc;
    {
        unique_lock<mutex> lock(mMutexCamera);
        Twc = mCameraPose.matrix();
    }

    for (int i = 0; i<4; i++) {
        M.m[4*i] = Twc(0,i);
        M.m[4*i+1] = Twc(1,i);
        M.m[4*i+2] = Twc(2,i);
        M.m[4*i+3] = Twc(3,i);
    }

    MOw.SetIdentity();
    MOw.m[12] = Twc(0,3);
    MOw.m[13] = Twc(1,3);
    MOw.m[14] = Twc(2,3);
}

#ifdef USE_DENSE_MAPPING
void MapDrawer::DrawOctoMap()
{
    //Map* pActiveMap = mpAtlas->GetCurrentMap();
    //if(!pActiveMap) return;
    //const vector<KeyFrame*> vKFs = pActiveMap->GetAllKeyFrames(); //  获取所有关键帧=====
    //int N = vKFs.size();// 当前关键帧数量======

    // 带有颜色的 octomap=====
    if(mpDenseMapper == NULL){
        std::cout<<"mpDenseMapper == NULL !!!!"<<std::endl;
        return;
    }
    int counter = 0;// 计数
    double occ_thresh = 0.8; // 概率阈值 原来 0.9  越大，显示的octomap格子越少
    // int level = 16; // 八叉树地图 深度???
    glClearColor(1.0f,1.0f,1.0f,1.0f);// 颜色 + 透明度

    glDisable(GL_LIGHTING);
    glEnable (GL_BLEND);
    int max_depth = 16;
    auto m_octree = mpDenseMapper->getOctreeCopy(); //no need to lock, by copy
    double minX, minY, minZ, maxX, maxY, maxZ;
    m_octree.getMetricMin(minX, minY, minZ);
    m_octree.getMetricMax(maxX, maxY, maxZ);
    for(auto it=m_octree.begin_tree(max_depth); it != m_octree.end_tree(); ++counter, ++it)
    {
       // if(level != it.getDepth()) continue;
       if(!it.isLeaf())continue;
       //double occ = it->getOccupancy();//占有概率=================
       //if(occ < occ_thresh)continue // 占有概率较低====不显示
       if(!m_octree.isNodeOccupied(*it))continue;
       // std::cout<< occ << std::endl;
       float halfsize = it.getSize()/2.0;// 半尺寸
       float x = it.getX();
       float y = it.getY();
       float z = it.getZ();

       // 高度====
       double h = ( std::min(std::max((y-minY)/(maxY-minY), 0.0), 1.0))*0.8;
       // 按高度 计算颜色
       double r, g, b;
       heightMapColor(h, r,g,b);

       glBegin(GL_TRIANGLES); // 三角形??
       //Front
       glColor3d(r, g, b);// 显示颜色=====
       glVertex3f(x-halfsize,y-halfsize,z-halfsize);// - - - 1
       glVertex3f(x-halfsize,y+halfsize,z-halfsize);// - + - 2
       glVertex3f(x+halfsize,y+halfsize,z-halfsize);// + + -3

       glVertex3f(x-halfsize,y-halfsize,z-halfsize); // - - -
       glVertex3f(x+halfsize,y+halfsize,z-halfsize); // + + -
       glVertex3f(x+halfsize,y-halfsize,z-halfsize); // + - -4

       //Back
       glVertex3f(x-halfsize,y-halfsize,z+halfsize); // - - + 1
       glVertex3f(x+halfsize,y-halfsize,z+halfsize); // + - + 2
       glVertex3f(x+halfsize,y+halfsize,z+halfsize); // + + + 3

       glVertex3f(x-halfsize,y-halfsize,z+halfsize); // - - +
       glVertex3f(x+halfsize,y+halfsize,z+halfsize); // + + +
       glVertex3f(x-halfsize,y+halfsize,z+halfsize); // - + + 4

       //Left
       glVertex3f(x-halfsize,y-halfsize,z-halfsize); // - - - 1
       glVertex3f(x-halfsize,y-halfsize,z+halfsize); // - - + 2
       glVertex3f(x-halfsize,y+halfsize,z+halfsize); // - + + 3

       glVertex3f(x-halfsize,y-halfsize,z-halfsize); // - - -
       glVertex3f(x-halfsize,y+halfsize,z+halfsize); // - + +
       glVertex3f(x-halfsize,y+halfsize,z-halfsize); // - + - 4

       //Right
       glVertex3f(x+halfsize,y-halfsize,z-halfsize);
       glVertex3f(x+halfsize,y+halfsize,z-halfsize);
       glVertex3f(x+halfsize,y+halfsize,z+halfsize);

       glVertex3f(x+halfsize,y-halfsize,z-halfsize);
       glVertex3f(x+halfsize,y+halfsize,z+halfsize);
       glVertex3f(x+halfsize,y-halfsize,z+halfsize);

       //top
       glVertex3f(x-halfsize,y-halfsize,z-halfsize);
       glVertex3f(x+halfsize,y-halfsize,z-halfsize);
       glVertex3f(x+halfsize,y-halfsize,z+halfsize);

       glVertex3f(x-halfsize,y-halfsize,z-halfsize);
       glVertex3f(x+halfsize,y-halfsize,z+halfsize);
       glVertex3f(x-halfsize,y-halfsize,z+halfsize);

       //bottom
       glVertex3f(x-halfsize,y+halfsize,z-halfsize);
       glVertex3f(x-halfsize,y+halfsize,z+halfsize);
       glVertex3f(x+halfsize,y+halfsize,z+halfsize);

       glVertex3f(x-halfsize,y+halfsize,z-halfsize);
       glVertex3f(x+halfsize,y+halfsize,z+halfsize);
       glVertex3f(x+halfsize,y+halfsize,z-halfsize);
       glEnd();

       glBegin(GL_LINES); // 线段=======
       glColor3f(0,0,0);
       //
       glVertex3f(x-halfsize,y-halfsize,z-halfsize);// - - - 1
       glVertex3f(x-halfsize,y+halfsize,z-halfsize);

       glVertex3f(x-halfsize,y+halfsize,z-halfsize);// - + - 2
       glVertex3f(x+halfsize,y+halfsize,z-halfsize);// + + -3

       glVertex3f(x+halfsize,y+halfsize,z-halfsize);// + + -3
       glVertex3f(x+halfsize,y-halfsize,z-halfsize); // + - -4

       glVertex3f(x+halfsize,y-halfsize,z-halfsize); // + - -4
       glVertex3f(x-halfsize,y-halfsize,z-halfsize);// - - - 1


       // back

       glVertex3f(x-halfsize,y-halfsize,z+halfsize); // - - + 1
       glVertex3f(x+halfsize,y-halfsize,z+halfsize); // + - + 2

       glVertex3f(x+halfsize,y-halfsize,z+halfsize); // + - + 2
       glVertex3f(x+halfsize,y+halfsize,z+halfsize); // + + + 3

       glVertex3f(x+halfsize,y+halfsize,z+halfsize); // + + + 3
       glVertex3f(x-halfsize,y+halfsize,z+halfsize); // - + + 4

       glVertex3f(x-halfsize,y+halfsize,z+halfsize); // - + + 4
       glVertex3f(x-halfsize,y-halfsize,z+halfsize); // - - + 1

       // top
       glVertex3f(x+halfsize,y-halfsize,z-halfsize);
       glVertex3f(x+halfsize,y-halfsize,z+halfsize);

       glVertex3f(x-halfsize,y-halfsize,z+halfsize);
       glVertex3f(x-halfsize,y-halfsize,z-halfsize);

        // bottom

       glVertex3f(x-halfsize,y+halfsize,z+halfsize);
       glVertex3f(x+halfsize,y+halfsize,z+halfsize);

       glVertex3f(x-halfsize,y+halfsize,z-halfsize);
       glVertex3f(x+halfsize,y+halfsize,z-halfsize);
       glEnd();
    }
    std::cout<<"counter: "<<counter<<std::endl;
}

// 按高度 绘制 octomap 颜色======
void MapDrawer::heightMapColor(double h, double& r, double &g, double& b)
{   
    double s = 1.0;
    double v = 1.0;

    h -= floor(h);
    h *= 6;

    int i;
    double m, n, f;

    i = floor(h);
    f = h - i;

    if(!(i & 1))
    {   
        f = 1 - f;
    }
    m = v * (1-s);
    n = v * (1- s*f);

    switch(i)
    {   
        case 6:
        case 0:
            r = v; g = n; b = m;
            break;
        case 1:
            r = n; g = v; b = m;
            break;
        case 2:
            r = m; g = v; b = n;
            break;
        case 3:
            r = m; g = n; b = v;
            break;
        case 4:
            r = n; g = m; b = v;
            break;
        case 5:
            r = v; g = m; b = n;
            break;
        default:
            r = 1; g = 0.5; b = 0.5;
         break;

    }

}
#endif

} //namespace ORB_SLAM
