TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt
DEFINES += GLIBCXX_USE_CXX11_ABI=0\
PROTOBUF_USE_DLLS

SOURCES += main.cpp \
    dasiamrpntracker.cpp

HEADERS += \
    dasiamrpntracker.h

INCLUDEPATH +=/home/dtt/pytorch/third_party/eigen
INCLUDEPATH +=/usr/local/ffmpeg/include\
/usr/include/opencv\
/usr/include/opencv2\
/usr/protobuf/include

unix:!macx: LIBS += -L$$PWD/../../../../usr/lib/nvidia-390/ -lGL

unix:!macx: LIBS += -L$$PWD/../../../../usr/lib/x86_64-linux-gnu/ -lxml2

unix:!macx: LIBS += -L$$PWD/../../../../usr/lib/x86_64-linux-gnu/ -lopencv_core

unix:!macx: LIBS += -L$$PWD/../../../../usr/lib/x86_64-linux-gnu/ -lopencv_highgui

unix:!macx: LIBS += -L$$PWD/../../../../usr/lib/x86_64-linux-gnu/ -lopencv_imgproc


#unix:!macx: LIBS += -L$$PWD/../../../../usr/lib/x86_64-linux-gnu/ -lopencv_imgcodecs

#unix:!macx: LIBS += -L$$PWD/../../../../usr/lib/x86_64-linux-gnu/ -lopencv_videoio


unix:!macx: LIBS += -L$$PWD/../../../../usr/lib/x86_64-linux-gnu/ -lpthread


unix:!macx: LIBS += -L$$PWD/../../../../usr/local/cuda/lib64/ -lcublas

unix:!macx: LIBS += -L$$PWD/../../../../usr/local/cuda/lib64/ -lcudart

unix:!macx: LIBS += -L$$PWD/../../../../usr/local/cuda/lib64/ -lcudnn

unix:!macx: LIBS += -L$$PWD/../../../../usr/local/cuda/lib64/ -lcufft

unix:!macx: LIBS += -L$$PWD/../../../../usr/local/cuda/lib64/ -lcurand

unix:!macx: LIBS += -L$$PWD/../../../../usr/local/cuda/lib64/ -lcuda

unix:!macx: LIBS += -L$$PWD/../../../../usr/lib/x86_64-linux-gnu/ -lglog

INCLUDEPATH += $$PWD/../../../../../../usr/lib/x86_64-linux-gnu \
/usr/local/cuda/include

unix:!macx: LIBS += -L$$PWD/../../../../usr/local/lib/ -lcaffe2_gpu

unix:!macx: LIBS += -L$$PWD/../../../../usr/local/lib/ -lcaffe2

unix:!macx: LIBS += -L$$PWD/../../../../usr/local/lib/ -lcaffe2_yolo_ops_gpu

unix:!macx: LIBS += -L$$PWD/../../../../usr/local/lib/ -lcaffe2_observers

unix:!macx: LIBS += -L$$PWD/../../../../usr/local/lib/ -lcaffe2_module_test_dynamic

unix:!macx: LIBS += -L$$PWD/../../../../../../usr/lib/x86_64-linux-gnu/ -ldl


unix:!macx: LIBS += -L$$PWD/../../../../../usr/lib/x86_64-linux-gnu/ -lgflags

unix:!macx: LIBS += -L$$PWD/../../../../../usr/lib/x86_64-linux-gnu/ -lglog



unix:!macx: LIBS += -L$$PWD/../../../../../usr/lib/x86_64-linux-gnu/ -lavcodec-ffmpeg

#unix:!macx: LIBS += -L$$PWD/../../../../../usr/lib/x86_64-linux-gnu/ -lavcodec
#unix:!macx: LIBS += -L$$PWD/../../ffmpeg-3.2.4/install/lib/ -lavcodec
unix:!macx: LIBS += -L$$PWD/../../../../usr/local/ffmpeg/lib/ -lavcodec

unix:!macx: LIBS += -L$$PWD/../../../../../usr/lib/x86_64-linux-gnu/ -lavformat-ffmpeg

#unix:!macx: LIBS += -L$$PWD/../../../../../usr/lib/x86_64-linux-gnu/ -lavformat
#unix:!macx: LIBS += -L$$PWD/../../ffmpeg-3.2.4/install/lib/ -lavformat
unix:!macx: LIBS += -L$$PWD/../../../../usr/local/ffmpeg/lib/ -lavformat

unix:!macx: LIBS += -L$$PWD/../../ -lavresample-ffmpeg

#unix:!macx: LIBS += -L$$PWD/../../../../../usr/lib/x86_64-linux-gnu/ -lavresample
unix:!macx: LIBS += -L$$PWD/../../ -lavresample

unix:!macx: LIBS += -L$$PWD/../../../../../usr/lib/x86_64-linux-gnu/ -lavutil-ffmpeg

#unix:!macx: LIBS += -L$$PWD/../../../../../usr/lib/x86_64-linux-gnu/ -lavutil
#unix:!macx: LIBS += -L$$PWD/../../ffmpeg-3.2.4/install/lib/ -lavutil
unix:!macx: LIBS += -L$$PWD/../../../../usr/local/ffmpeg/lib/ -lavutil

unix:!macx: LIBS += -L$$PWD/../../../../../usr/lib/x86_64-linux-gnu/ -lswresample-ffmpeg

#unix:!macx: LIBS += -L$$PWD/../../../../../usr/lib/x86_64-linux-gnu/ -lswresample
#unix:!macx: LIBS += -L$$PWD/../../ffmpeg-3.2.4/install/lib/ -lswresample
unix:!macx: LIBS += -L$$PWD/../../../../usr/local/ffmpeg/lib/ -lswresample

#unix:!macx: LIBS += -L$$PWD/../../../../../usr/lib/x86_64-linux-gnu/ -lswscale
#unix:!macx: LIBS += -L$$PWD/../../ffmpeg-3.2.4/install/lib/ -lswscale
unix:!macx: LIBS += -L$$PWD/../../../../usr/local/ffmpeg/lib/ -lswscale

#unix:!macx: LIBS += -L$$PWD/../HBGS_IVA_CarAnalysisServer-qt-ice364-stream-from-yushi-sdk/bin/ -lclntsh


unix:!macx: LIBS += -L$$PWD/../../../../usr/protobuf/lib/ -lprotobuf-lite

unix:!macx: LIBS += -L$$PWD/../../../../usr/protobuf/lib/ -lprotobuf

