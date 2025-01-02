cmake_minimum_required(VERSION 3.5)

project(example LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(FFMEPG_HEADERS_DIR /usr/include)
set(FFMPEG_LIBS_DIR /usr/lib)
include_directories(${FFMEPG_HEADERS_DIR})
link_directories(${FFMPEG_LIBS_DIR})
set(FFMPEG_LIBS libavcodec.so libavformat.so libswscale.so libavdevice.so libavutil.so)

include_directories("/home/lotuscc/gitlib/tensorrt/TensorRT-8.6.1.6/include/")
link_directories("/home/lotuscc/gitlib/tensorrt/TensorRT-8.6.1.6/lib")

include_directories("/opt/cuda/include/")
link_directories("/opt/cuda/lib64")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(TENSORRT_LIB_PATH "/home/lotuscc/gitlib/tensorrt/TensorRT-8.6.1.6/lib")
set(CUDA_LIB_PATH "/opt/cuda/lib64")

file(GLOB TensorRT_LIBS "${TENSORRT_LIB_PATH}/*.so")
file(GLOB CUDA_LIBS "${CUDA_LIB_PATH}/*.so")

include_directories("./context/")
include_directories("./model/")
include_directories("./tools/")
# include_directories("./example/")

# include_directories("/usr/include")


add_executable(example main.cpp    
    ffmpeg.cpp    
    globalComm.cpp
    context/baseContext.h context/contextDoer_handPoseX.h context/contextDoer_yolov8Detect.h model/baseModel.h model/model_handPoseX.h model/model_yolov8Detect.h model/templateModel.h
    context/contextDoer_yolov8Pose.h
    tools/CustomSink.h tools/ffmpeg.h tools/globalComm.h tools/preProcess.h

    example/example_yolov8Detect.h
    example/example_handPoseX.h
    example/example_yolov8Pose.h
    context/contextDoer_yolov8Cls.h
    example/example_yolov8Cls.h
    context/contextDoer_scrfd.h
    example/example_scrfd.h
    context/contextDoer_arcFace.h
    example/example_arcFace.h
    example/example_ffmpeg.h
    tools/face_align.h
    tools/facePreprocess.h
    tools/ffmpgeDecoderWithCPU.h
    tools/ffmpgeDecoderWithGPU.h
    context/contextDoer_yolov8Seg.h
    example/example_yolov8Seg.h
    context/baseTensor.h
    context/macro.h
)

include(GNUInstallDirs)
install(TARGETS example
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})

set(BUILD_SHARED_LIBS ON)

add_library(cudalibx SHARED
    hello.h
    hello.cu
    imageProcess.cu
    imageProcess.h)
target_compile_features(cudalibx PUBLIC cxx_std_11)


find_package(Boost 1.83.0 COMPONENTS system thread)
include_directories(${Boost_INCLUDE_DIRS})


# set(Boost_USE_STATIC_LIBS OFF)
# set(Boost_USE_MULTITHREADED ON)
# set(Boost_USE_STATIC_RUNTIME OFF)
# find_package(Boost 1.83.0 COMPONENTS *boost libraries here*)
# if(Boost_FOUND)
#     include_directories(${Boost_INCLUDE_DIRS})
#     # add_executable(progname file1.cxx file2.cxx)
#     target_link_libraries(progname ${Boost_LIBRARIES})
# endif()

find_package(g3log CONFIG REQUIRED)
find_package(OpenCV 4  REQUIRED)

# PRIVATE

target_link_libraries(example PRIVATE g3log ${FFMPEG_LIBS} ${TensorRT_LIBS} ${CUDA_LIBS} ${Boost_LIBRARIES} opencv_highgui opencv_core opencv_imgproc opencv_dnn cudalibx)

# target_link_libraries(example ${FFMPEG_LIBS})
# target_link_libraries(example opencv_highgui opencv_core opencv_imgproc)
# target_link_libraries(example ${OpenCV_LIBS})
# target_link_libraries(example ${Boost_LIBRARIES})


MESSAGE( STATUS "this FFMPEG_LIBS var key = ${FFMPEG_LIBS}")
MESSAGE( STATUS "this TensorRT_LIBS var key = ${TensorRT_LIBS}")
MESSAGE( STATUS "this CUDA_LIBS var key = ${CUDA_LIBS}")
MESSAGE( STATUS "this Boost_LIBRARIES var key = ${Boost_LIBRARIES}")

