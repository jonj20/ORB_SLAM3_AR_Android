#include <jni.h>
#include <string>
#include "System.h"
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include "../a2ir/phone_sensor.h"
#include "../a2ir/timer.h"
#include <time.h>
#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

#include "Process.h"
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>


//orbslam2
#include <opencv2/opencv.hpp>
//#include "include/System.h"
#include "Common.h"
#include "Plane.h"
#include "UIUtils.h"
#include "Matrix.h"

//#define SLAM_USE_IMU_MONOCULAR


extern "C" {

std::string modelPath;

//ORB_SLAM3::System *slamSys;
Plane *pPlane;
float fx,fy,cx,cy;
//double timeStamp;
bool slamInited = false;
#ifdef SLAM_USE_IMU_MONOCULAR
a2ir::PhoneSensor *phoneSensorInstance = NULL;
#endif
vector<ORB_SLAM3::MapPoint*> vMPs;
vector<cv::KeyPoint> vKeys;
cv::Mat Tcw;
//Sophus::SE3f Tcw;
const int PLANE_DETECTED = 233;
const int PLANE_NOT_DETECTED = 1234;

const bool DRAW_STATUS = false;

//#define ASSETS_PATH "/storage/emulated/0/SLAM"
#define ASSETS_PATH "/storage/emulated/0/Android/data/com.martin.ads.slamar3/files/SLAM"

//slam2-3
#define BOWISBIN

#ifdef BOWISBIN
ORB_SLAM3::System* slamSys = nullptr;
#endif
std::chrono::steady_clock::time_point t0;
double tframe = 0;

cv::Mat Plane2World = cv::Mat::eye(4,4,CV_32F);
cv::Mat Marker2World = cv::Mat::eye(4,4,CV_32F);
cv::Mat centroid;

bool load_as_text(ORB_SLAM3::ORBVocabulary* voc, const std::string infile) {
    clock_t tStart = clock();
    bool res = voc->loadFromTextFile(infile);
    LOGE("Loading fom text: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
    return res;
}

void save_as_binary(ORB_SLAM3::ORBVocabulary* voc, const std::string outfile) {
    clock_t tStart = clock();
    voc->saveToBinaryFile(outfile);
    LOGE("Saving as binary: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
}

void txt_2_bin(){
    ORB_SLAM3::ORBVocabulary* voc = new ORB_SLAM3::ORBVocabulary();

    load_as_text(voc, ASSETS_PATH "/ORBvoc.txt");
    save_as_binary(voc, ASSETS_PATH "/ORBvoc.txt.arm.bin");
}

cv::Point2f Camera2Pixel(cv::Mat poseCamera,cv::Mat mk){
    return Point2f(
            poseCamera.at<float>(0,0)/poseCamera.at<float>(2,0)*mk.at<float>(0,0)+mk.at<float>(0,2),
            poseCamera.at<float>(1,0)/poseCamera.at<float>(2,0)*mk.at<float>(1,1)+mk.at<float>(1,2)
    );
}

#ifdef SLAM_USE_IMU_MONOCULAR
std::mutex m_buf;
queue<ORB_SLAM3::IMU::Point> imu_buf;
void clearImus() {
    m_buf.lock();
    if (imu_buf.size() > 100) { //100 -capacity
        imu_buf.pop();
    }
    m_buf.unlock();
}
std::vector<ORB_SLAM3::IMU::Point> getValidImus(double imgTimestamp) {
    std::vector<ORB_SLAM3::IMU::Point> IMUs;
   
   // m_buf.lock();
        while ( (!imu_buf.empty()) && (imu_buf.front().t < imgTimestamp)) {
            IMUs.emplace_back(imu_buf.front());
            imu_buf.pop();
        }
  //  m_buf.lock();
   
    return IMUs;
}
#endif

void receiveImu(IMU_MSG acc_gro){
#ifdef SLAM_USE_IMU_MONOCULAR
    ORB_SLAM3::IMU::Point point = ORB_SLAM3::IMU::Point(
        acc_gro.gyr[0],
        acc_gro.gyr[1],
        acc_gro.gyr[2],
        acc_gro.acc[0],
        acc_gro.acc[1],
        acc_gro.acc[2],
        acc_gro.header
    );
    m_buf.lock();
    imu_buf.push(point);
    if (imu_buf.size() >= 100) { //100 -capacity
        imu_buf.pop();
    }
    m_buf.unlock();
#endif
}

int processImage(cv::Mat &image,cv::Mat &outputImage,int statusBuf[]){
    //timeStamp+=1/30.0;
    //Martin:about 140-700ms to finish last part
    //LOGD("Started.");
    cv::Mat imgSmall;
    cv::resize(image,imgSmall,cv::Size(image.cols/2,image.rows/2));
    double imgTimestamp = PVR::Timer::GetSeconds();
#ifdef SLAM_USE_IMU_MONOCULAR
    vector<ORB_SLAM3::IMU::Point> vImuMeas;
    // Load imu measurements from previous frame
    vImuMeas.clear();
    vImuMeas = getValidImus(imgTimestamp);
    LOGD("vImuMeas.size %d",vImuMeas.size());
    Sophus::SE3f Tcw_SE3f = slamSys->TrackMonocular(imgSmall,imgTimestamp, vImuMeas);
#else
    Sophus::SE3f Tcw_SE3f = slamSys->TrackMonocular(imgSmall,imgTimestamp);
#endif
    Eigen::Matrix4f Tcw_Matrix = Tcw_SE3f.matrix();
    cv::eigen2cv(Tcw_Matrix, Tcw);

    //LOGD("finished. %d %d",imgSmall.rows,imgSmall.cols);
    //Martin:about 5ms to finish last part
    //Martin:about 5-10ms to await next image
    int status = slamSys->GetTrackingState();
    vMPs = slamSys->GetTrackedMapPoints();
    vKeys = slamSys->GetTrackedKeyPointsUn();
    if(DRAW_STATUS)
        printStatus(status,outputImage);
    drawTrackedPoints(vKeys,vMPs,outputImage);

    LOGD("processImage: status: %d,vp count: %d", status, vMPs.size());
    //cv::imwrite(modelPath+"/lala2.jpg",outputImage);
    if(status==2) {
        if(DRAW_STATUS){
            char buf[22];
            if(pPlane){
                sprintf(buf,"Plane Detected");
                cv::putText(outputImage,buf,cv::Point(10,30),cv::FONT_HERSHEY_PLAIN,1.5,cv::Scalar(255,0,255),2,8);
            }
            else {
                sprintf(buf,"Plane not detected.");
                cv::putText(outputImage,buf,cv::Point(10,30),cv::FONT_HERSHEY_PLAIN,1.5,cv::Scalar(255,255,0),2,8);
            }
        }
    }
    return status;
}

JNIEXPORT void JNICALL
Java_com_martin_ads_slamar3_NativeHelper_initSLAM(JNIEnv *env, jobject instance,
                                                               jstring path_) {
    const char *path = env->GetStringUTFChars(path_, 0);

    if(slamInited) return;
    slamInited=true;
    modelPath=path;
    LOGE("Model Path is %s",modelPath.c_str());
    env->ReleaseStringUTFChars(path_, path);
    //txt_2_bin(); //test
    ifstream in;
    in.open(modelPath+"config.txt");
    string vocName,settingName;
    getline(in,vocName);
    getline(in,settingName);
    vocName=modelPath+vocName;
    settingName=modelPath+settingName;

    cv::FileStorage fSettings(settingName, cv::FileStorage::READ);
    fx = fSettings["Camera1.fx"];
    fy = fSettings["Camera1.fy"];
    cx = fSettings["Camera1.cx"];
    cy = fSettings["Camera1.cy"];

    //timeStamp=0;
    LOGD("%s %c %s %c",vocName.c_str(),vocName[vocName.length()-1],settingName.c_str(),settingName[settingName.length()-1]);
#ifdef SLAM_USE_IMU_MONOCULAR
    slamSys = new ORB_SLAM3::System(vocName, settingName, ORB_SLAM3::System::IMU_MONOCULAR, false);
    //IMU
    clearImus();
    phoneSensorInstance = a2ir::PhoneSensor::GetInstance();
    if (NULL != phoneSensorInstance)
        phoneSensorInstance->StartTrack();
#else
    slamSys = new ORB_SLAM3::System(vocName, settingName, ORB_SLAM3::System::MONOCULAR, false);
#endif
}

JNIEXPORT void JNICALL
Java_com_martin_ads_slamar3_NativeHelper_nativeProcessFrameMat(JNIEnv *env, jobject instance,
                                                              jlong matAddrGr, jlong matAddrRgba,
                                                              jintArray statusBuf_) {
    jint *statusBuf = env->GetIntArrayElements(statusBuf_, NULL);

    cv::Mat &mGr = *(cv::Mat *) matAddrGr;
    cv::Mat &mRgba = *(cv::Mat *) matAddrRgba;
    statusBuf[0]=processImage(mGr,mRgba,statusBuf);

    env->ReleaseIntArrayElements(statusBuf_, statusBuf, 0);
}
JNIEXPORT void JNICALL
Java_com_martin_ads_slamar3_NativeHelper_detect(JNIEnv *env, jobject instance,
                                               jintArray statusBuf_) {
    jint *statusBuf = env->GetIntArrayElements(statusBuf_, NULL);
    if(!Tcw.empty()){
        pPlane=detectPlane(Tcw,vMPs,50);
        if(pPlane && slamSys->MapChanged())
            pPlane->Recompute();
        statusBuf[1]=pPlane? PLANE_DETECTED:PLANE_NOT_DETECTED;
    }
    env->ReleaseIntArrayElements(statusBuf_, statusBuf, 0);
}

JNIEXPORT void JNICALL
Java_com_martin_ads_slamar3_NativeHelper_getM(JNIEnv *env, jobject instance, jfloatArray modelM_) {
    jfloat *modelM = env->GetFloatArrayElements(modelM_, NULL);

    getRUBModelMatrixFromRDF(pPlane->glTpw,modelM);

    env->ReleaseFloatArrayElements(modelM_, modelM, 0);
}
JNIEXPORT void JNICALL
Java_com_martin_ads_slamar3_NativeHelper_getV(JNIEnv *env, jobject instance, jfloatArray viewM_) {
    jfloat *viewM = env->GetFloatArrayElements(viewM_, NULL);

    float tmpM[16];
    getColMajorMatrixFromMat(tmpM,Tcw);
    getRUBViewMatrixFromRDF(tmpM,viewM);

    env->ReleaseFloatArrayElements(viewM_, viewM, 0);
}
JNIEXPORT void JNICALL
Java_com_martin_ads_slamar3_NativeHelper_getP(JNIEnv *env, jobject instance, jint imageWidth,
                                             jint imageHeight, jfloatArray projectionM_) {
    jfloat *projectionM = env->GetFloatArrayElements(projectionM_, NULL);

    //TODO:/2
    frustumM_RUB(imageWidth/2,imageHeight/2,fx,fy,cx,cy,0.1,1000,projectionM);

    env->ReleaseFloatArrayElements(projectionM_, projectionM, 0);
}
}

