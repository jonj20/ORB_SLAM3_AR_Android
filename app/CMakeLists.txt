cmake_minimum_required(VERSION 3.4.1)

SET(CMAKE_BUILD_TYPE Release)
#SET(CMAKE_BUILD_TYPE Debug)

# # pcl 
# find_package(PCL REQUIRED)
# include_directories(${PCL_INCLUDE_DIRS})
# add_definitions(${PCL_DEFINITIONS})

# # octomap 
# find_package(octomap REQUIRED)
# include_directories(${OCTOMAP_INCLUDE_DIRS})

include_directories(
	${PROJECT_SOURCE_DIR}/src/main/cpp
	${PROJECT_SOURCE_DIR}/src/main/cpp/include
	${PROJECT_SOURCE_DIR}/src/main/cpp/include/CameraModels
	${PROJECT_SOURCE_DIR}/src/main/cpp/Thirdparty/Sophus
	${PROJECT_SOURCE_DIR}/src/main/cpp/Thirdparty/Sophus/sophus
	${PROJECT_SOURCE_DIR}/src/main/cpp/Thirdparty/Eigen/eigen-3.4.0
	${PROJECT_SOURCE_DIR}/src/main/cpp/Thirdparty/Eigen/eigen-3.4.0/src
	${PROJECT_SOURCE_DIR}/src/main/cpp/Thirdparty/boost/boost-1_72_0
	${PROJECT_SOURCE_DIR}/src/main/cpp/Thirdparty/boost/boost-1_72_0/boost
	${PROJECT_SOURCE_DIR}/src/main/cpp/Thirdparty/openssl/openssl-1.0.2s
	${PROJECT_SOURCE_DIR}/src/main/cpp/Thirdparty/openssl/openssl-1.0.2s/openssl
	${PROJECT_SOURCE_DIR}/src/main/cpp/opencv/opencv-4.5.5/include
	${PROJECT_SOURCE_DIR}/src/main/cpp/src/a2ir		 
)



include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)


# set opencv shared library
add_library(libopencv_java4 SHARED IMPORTED)
set_target_properties(libopencv_java4 PROPERTIES IMPORTED_LOCATION
		${CMAKE_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI}/libopencv_java4.so)



set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -frtti -fopenmp -static-openmp")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -frtti -fopenmp -static-openmp")


ADD_LIBRARY(g2o
		src/main/cpp/Thirdparty/g2o/g2o/core/base_binary_edge.h
		src/main/cpp/Thirdparty/g2o/g2o/core/base_binary_edge.hpp
		src/main/cpp/Thirdparty/g2o/g2o/core/base_edge.h
		src/main/cpp/Thirdparty/g2o/g2o/core/base_multi_edge.h
		src/main/cpp/Thirdparty/g2o/g2o/core/base_multi_edge.hpp
		src/main/cpp/Thirdparty/g2o/g2o/core/base_unary_edge.h
		src/main/cpp/Thirdparty/g2o/g2o/core/base_unary_edge.hpp
		src/main/cpp/Thirdparty/g2o/g2o/core/base_vertex.h
		src/main/cpp/Thirdparty/g2o/g2o/core/base_vertex.hpp
		src/main/cpp/Thirdparty/g2o/g2o/core/batch_stats.cpp
		src/main/cpp/Thirdparty/g2o/g2o/core/batch_stats.h
		src/main/cpp/Thirdparty/g2o/g2o/core/block_solver.h
		src/main/cpp/Thirdparty/g2o/g2o/core/block_solver.hpp
		src/main/cpp/Thirdparty/g2o/g2o/core/cache.cpp
		src/main/cpp/Thirdparty/g2o/g2o/core/cache.h
		src/main/cpp/Thirdparty/g2o/g2o/core/creators.h
		src/main/cpp/Thirdparty/g2o/g2o/core/eigen_types.h
		src/main/cpp/Thirdparty/g2o/g2o/core/estimate_propagator.cpp
		src/main/cpp/Thirdparty/g2o/g2o/core/estimate_propagator.h
		src/main/cpp/Thirdparty/g2o/g2o/core/factory.cpp
		src/main/cpp/Thirdparty/g2o/g2o/core/factory.h
		src/main/cpp/Thirdparty/g2o/g2o/core/hyper_dijkstra.cpp
		src/main/cpp/Thirdparty/g2o/g2o/core/hyper_dijkstra.h
		src/main/cpp/Thirdparty/g2o/g2o/core/hyper_graph.cpp
		src/main/cpp/Thirdparty/g2o/g2o/core/hyper_graph.h
		src/main/cpp/Thirdparty/g2o/g2o/core/hyper_graph_action.cpp
		src/main/cpp/Thirdparty/g2o/g2o/core/hyper_graph_action.h
		src/main/cpp/Thirdparty/g2o/g2o/core/jacobian_workspace.cpp
		src/main/cpp/Thirdparty/g2o/g2o/core/jacobian_workspace.h
		src/main/cpp/Thirdparty/g2o/g2o/core/linear_solver.h
		src/main/cpp/Thirdparty/g2o/g2o/core/marginal_covariance_cholesky.cpp
		src/main/cpp/Thirdparty/g2o/g2o/core/marginal_covariance_cholesky.h
		src/main/cpp/Thirdparty/g2o/g2o/core/matrix_operations.h
		src/main/cpp/Thirdparty/g2o/g2o/core/matrix_structure.cpp
		src/main/cpp/Thirdparty/g2o/g2o/core/matrix_structure.h
		src/main/cpp/Thirdparty/g2o/g2o/core/openmp_mutex.h
		src/main/cpp/Thirdparty/g2o/g2o/core/optimizable_graph.cpp
		src/main/cpp/Thirdparty/g2o/g2o/core/optimizable_graph.h
		src/main/cpp/Thirdparty/g2o/g2o/core/optimization_algorithm.cpp
		src/main/cpp/Thirdparty/g2o/g2o/core/optimization_algorithm.h
		src/main/cpp/Thirdparty/g2o/g2o/core/optimization_algorithm_dogleg.cpp
		src/main/cpp/Thirdparty/g2o/g2o/core/optimization_algorithm_dogleg.h
		src/main/cpp/Thirdparty/g2o/g2o/core/optimization_algorithm_factory.cpp
		src/main/cpp/Thirdparty/g2o/g2o/core/optimization_algorithm_factory.h
		src/main/cpp/Thirdparty/g2o/g2o/core/optimization_algorithm_gauss_newton.cpp
		src/main/cpp/Thirdparty/g2o/g2o/core/optimization_algorithm_gauss_newton.h
		src/main/cpp/Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.cpp
		src/main/cpp/Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h
		src/main/cpp/Thirdparty/g2o/g2o/core/optimization_algorithm_property.h
		src/main/cpp/Thirdparty/g2o/g2o/core/optimization_algorithm_with_hessian.cpp
		src/main/cpp/Thirdparty/g2o/g2o/core/optimization_algorithm_with_hessian.h
		src/main/cpp/Thirdparty/g2o/g2o/core/parameter.cpp
		src/main/cpp/Thirdparty/g2o/g2o/core/parameter.h
		src/main/cpp/Thirdparty/g2o/g2o/core/parameter_container.cpp
		src/main/cpp/Thirdparty/g2o/g2o/core/parameter_container.h
		src/main/cpp/Thirdparty/g2o/g2o/core/robust_kernel.cpp
		src/main/cpp/Thirdparty/g2o/g2o/core/robust_kernel.h
		src/main/cpp/Thirdparty/g2o/g2o/core/robust_kernel_factory.cpp
		src/main/cpp/Thirdparty/g2o/g2o/core/robust_kernel_factory.h
		src/main/cpp/Thirdparty/g2o/g2o/core/robust_kernel_impl.cpp
		src/main/cpp/Thirdparty/g2o/g2o/core/robust_kernel_impl.h
		src/main/cpp/Thirdparty/g2o/g2o/core/solver.cpp
		src/main/cpp/Thirdparty/g2o/g2o/core/solver.h
		src/main/cpp/Thirdparty/g2o/g2o/core/sparse_block_matrix.h
		src/main/cpp/Thirdparty/g2o/g2o/core/sparse_block_matrix.hpp
		src/main/cpp/Thirdparty/g2o/g2o/core/sparse_block_matrix_ccs.h
		src/main/cpp/Thirdparty/g2o/g2o/core/sparse_block_matrix_diagonal.h
		src/main/cpp/Thirdparty/g2o/g2o/core/sparse_block_matrix_test.cpp
		src/main/cpp/Thirdparty/g2o/g2o/core/sparse_optimizer.cpp
		src/main/cpp/Thirdparty/g2o/g2o/core/sparse_optimizer.h
		src/main/cpp/Thirdparty/g2o/g2o/solvers/linear_solver_dense.h
		src/main/cpp/Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h
		src/main/cpp/Thirdparty/g2o/g2o/stuff/color_macros.h
		src/main/cpp/Thirdparty/g2o/g2o/stuff/macros.h
		src/main/cpp/Thirdparty/g2o/g2o/stuff/misc.h
		src/main/cpp/Thirdparty/g2o/g2o/stuff/os_specific.c
		src/main/cpp/Thirdparty/g2o/g2o/stuff/os_specific.h
		src/main/cpp/Thirdparty/g2o/g2o/stuff/property.cpp
		src/main/cpp/Thirdparty/g2o/g2o/stuff/property.h
		src/main/cpp/Thirdparty/g2o/g2o/stuff/string_tools.cpp
		src/main/cpp/Thirdparty/g2o/g2o/stuff/string_tools.h
		src/main/cpp/Thirdparty/g2o/g2o/stuff/timeutil.cpp
		src/main/cpp/Thirdparty/g2o/g2o/stuff/timeutil.h
		src/main/cpp/Thirdparty/g2o/g2o/types/se3_ops.h
		src/main/cpp/Thirdparty/g2o/g2o/types/se3_ops.hpp
		src/main/cpp/Thirdparty/g2o/g2o/types/se3mat.cpp
		src/main/cpp/Thirdparty/g2o/g2o/types/se3mat.h
		src/main/cpp/Thirdparty/g2o/g2o/types/se3quat.h
		src/main/cpp/Thirdparty/g2o/g2o/types/sim3.h
		src/main/cpp/Thirdparty/g2o/g2o/types/types_sba.cpp
		src/main/cpp/Thirdparty/g2o/g2o/types/types_sba.h
		src/main/cpp/Thirdparty/g2o/g2o/types/types_seven_dof_expmap.cpp
		src/main/cpp/Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h
		src/main/cpp/Thirdparty/g2o/g2o/types/types_six_dof_expmap.cpp
		src/main/cpp/Thirdparty/g2o/g2o/types/types_six_dof_expmap.h
		)

ADD_LIBRARY(DBoW2
        src/main/cpp/Thirdparty/DBoW2/DBoW2/BowVector.cpp
        src/main/cpp/Thirdparty/DBoW2/DBoW2/BowVector.h
        src/main/cpp/Thirdparty/DBoW2/DBoW2/FClass.h
        src/main/cpp/Thirdparty/DBoW2/DBoW2/FORB.cpp
        src/main/cpp/Thirdparty/DBoW2/DBoW2/FORB.h
        src/main/cpp/Thirdparty/DBoW2/DBoW2/FeatureVector.cpp
        src/main/cpp/Thirdparty/DBoW2/DBoW2/FeatureVector.h
        src/main/cpp/Thirdparty/DBoW2/DBoW2/ScoringObject.cpp
        src/main/cpp/Thirdparty/DBoW2/DBoW2/ScoringObject.h
        src/main/cpp/Thirdparty/DBoW2/DBoW2/TemplatedVocabulary.h
        src/main/cpp/Thirdparty/DBoW2/DUtils/Random.cpp
        src/main/cpp/Thirdparty/DBoW2/DUtils/Random.h
        src/main/cpp/Thirdparty/DBoW2/DUtils/Timestamp.cpp
        src/main/cpp/Thirdparty/DBoW2/DUtils/Timestamp.h
)

add_library(ORBSLAM3
    src/main/cpp/src/Atlas.cc
    src/main/cpp/src/CameraModels/KannalaBrandt8.cpp
    src/main/cpp/src/CameraModels/Pinhole.cpp
    src/main/cpp/src/Config.cc
    src/main/cpp/src/Converter.cc
    src/main/cpp/src/Frame.cc
    src/main/cpp/src/FrameDrawer.cc
    src/main/cpp/src/G2oTypes.cc
    src/main/cpp/src/GeometricTools.cc
    src/main/cpp/src/ImuTypes.cc
    src/main/cpp/src/KeyFrame.cc
    src/main/cpp/src/KeyFrameDatabase.cc
    src/main/cpp/src/LocalMapping.cc
    src/main/cpp/src/LoopClosing.cc
    src/main/cpp/src/MLPnPsolver.cpp
    src/main/cpp/src/Map.cc
    #src/main/cpp/src/MapDrawer.cc
    src/main/cpp/src/MapPoint.cc
    src/main/cpp/src/ORBextractor.cc
    src/main/cpp/src/ORBmatcher.cc
    src/main/cpp/src/OptimizableTypes.cpp
    src/main/cpp/src/Optimizer.cc
    src/main/cpp/src/Settings.cc
    src/main/cpp/src/Sim3Solver.cc
    src/main/cpp/src/System.cc
    src/main/cpp/src/Tracking.cc
    src/main/cpp/src/TwoViewReconstruction.cc
    #src/main/cpp/src/Viewer.cc
)

#C++日志
find_library( log-lib log )
find_library( android-lib android )


add_library(
        SLAM_JNI

        SHARED

	#imu sensor
        src/main/cpp/src/a2ir/log_util.cpp
        src/main/cpp/src/a2ir/phone_sensor.cpp
        src/main/cpp/src/a2ir/timer.cpp

        src/main/cpp/native-lib.cpp
        src/main/cpp/UIUtils.cpp
        src/main/cpp/Plane.cpp
        src/main/cpp/Matrix.cpp
        src/main/cpp/Frame.cpp
        src/main/cpp/Marker.cpp
        src/main/cpp/Process.cpp
        src/main/cpp/Camera.cpp
		#src/main/cpp/DenseMapping.cpp
 )

#以导入方式添加静态库
add_library(boost_system STATIC IMPORTED)
#指定导入的库位置
set_target_properties(boost_system PROPERTIES IMPORTED_LOCATION
${CMAKE_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI}/libboost_1_72_0/libboost_system.a)

#以导入方式添加静态库
add_library(boost_atomic STATIC IMPORTED)
#指定导入的库位置
set_target_properties(boost_atomic PROPERTIES IMPORTED_LOCATION
${CMAKE_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI}/libboost_1_72_0/libboost_atomic.a)

#以导入方式添加静态库
add_library(boost_serialization STATIC IMPORTED)
#指定导入的库位置
set_target_properties(boost_serialization PROPERTIES IMPORTED_LOCATION
${CMAKE_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI}/libboost_1_72_0/libboost_serialization.a)

add_library(ssl STATIC IMPORTED)
set_target_properties(ssl
		PROPERTIES IMPORTED_LOCATION
		${CMAKE_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI}/libopenssl-1.0.2s/libssl.a)

add_library(crypto STATIC IMPORTED)
set_target_properties(crypto
		PROPERTIES IMPORTED_LOCATION
		${CMAKE_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI}/libopenssl-1.0.2s/libcrypto.a)

target_link_libraries(ORBSLAM3
		DBoW2
		g2o
		${EIGEN3_LIBS}
		${SOPHUS_LIBS}
		# ${OpenCV_LIBS}
		boost_system
		boost_serialization
		ssl
		crypto
		)



target_link_libraries( # Specifies the target library.
                       SLAM_JNI
                       # Links the target library to the log library
                       # included in the NDK.
		       ${android-lib}
                       ${log-lib}
                       # Links the target library to the ORBSLAM3 library
                       ORBSLAM3
                       EGL
                       GLESv1_CM
                       # Links the target library to opencv shared libs
                       libopencv_java4
		)
