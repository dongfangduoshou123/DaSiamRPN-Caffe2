# DaSiamRPN-Caffe2
Distractor-aware Siamese Networks for Visual Object Tracking Inference in C++ powered by Caffe2

For more detail information:Please visit the original Pytorch repo:https://github.com/foolwood/DaSiamRPN

    python2.7  (for convert model only)
    caffe2
    Eigen
    opencv
    CUDA
The Model is not Convert by the ONNX but by a simple python script(Since the DaSiamRPN's inference is a Dynamic graph Not Static graph.As description in onnx it has limit to export Dynamic graph.)

Converted model for SiamRPN's Download link:


Other note: This Repo uses the older Caffe2,the Newest Caffe2 may be not work well(For Example the newest caffe2's Tensor class has been removed the template etc.).
One of the Old Caffe2 can be clone from https://github.com/zj463261929/caffe2_old



