#include <iostream>
#include "dasiamrpntracker.h"
using namespace std;

int main(int argc, char *argv[])
{
    FLAGS_alsologtostderr = 1;
    if (access("logs", F_OK) == -1) {
        system("mkdir logs");
    }
    FLAGS_log_dir = "./logs";
    google::InitGoogleLogging((argv)[0]);
    caffe2::GlobalInit(&argc, &argv);

    cv::VideoCapture cap;
    cap.open(0);
    if(cap.isOpened()){
        std::cout << "open camera ok";
    }
    BoxInfo binfo;
    binfo.xc = 150;
    binfo.yc = 150;
    binfo.h = 200;
    binfo.w = 200;
    binfo.best_score =0;
    cv::Mat mat;
    while(true){
        cap >> mat;
        int k = cv::waitKey(50);
//        mat = cv::imread("/opt/tou.jpg");
        cv::rectangle(mat,
                      cv::Point(int(binfo.xc-binfo.w/2),int(binfo.yc - binfo.h/2)),
                      cv::Point(int(binfo.xc + binfo.w/2),int(binfo.yc + binfo.h/2)),
                      1,8,0);
        cv::imshow("tmp",mat);
        if(k == 27)
            break;

    }
    TrackInfo info;       
    info.binfo = binfo;
    DaSiamRPNTracker tracker = DaSiamRPNTracker();
    tracker.SiamRPN_init(mat,binfo,info,global_init_net_file,temple_net_file,track_net_file,caffe2::DeviceType::CUDA,0);
    while(true){
        cap >> mat;
//        mat = cv::imread("/opt/tou.jpg");
        tracker.SiamRPN_track(mat,info);
        float x = info.binfo.xc-info.binfo.w/2;
        float y = info.binfo.yc-info.binfo.h/2;
        float xmax = info.binfo.xc + info.binfo.w/2;
        float ymax = info.binfo.yc + info.binfo.h/2;
//        LOG(INFO) << x << " " << y << " " << xmax << " " << ymax << std::endl;
        LOG(INFO) << info.binfo.best_score;
//        cv::rectangle(mat,
//                      cv::Point(int(x),int(y)),
//                      cv::Point(int(xmax),int(ymax)),
//                      1,8,0);
//        cv::imshow("tmp",mat);
//        int k = cv::waitKey(50);
//        if(k == 27)
//            break;
    }
    return 0;
}
