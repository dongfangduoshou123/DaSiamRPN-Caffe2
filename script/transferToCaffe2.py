# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# -------------------------------------------------------
#transfor to caffe2 part added by wzq 2018-10
# --------------------------------------------------------
import torch.nn as nn
import torch.nn.functional as F
import torch
import cv2

from caffe2.python import core
from caffe2.python import workspace
from caffe2.python import model_helper,brew
from caffe2.python import predictor
from caffe2.python.predictor.predictor_exporter import *

import numpy as np
import exportaspb
from caffe2.python import dyndep
cnn_arg_scope = {
            'order': "NCHW",
            'use_cudnn': False,
            'cudnn_exhaustive_search': False,
}

class SiamRPNBIG(nn.Module):
    def __init__(self, feat_in=512, feature_out=512, anchor=5):
        super(SiamRPNBIG, self).__init__()
        self.anchor = anchor
        self.feature_out = feature_out
        self.featureExtract = nn.Sequential(
            nn.Conv2d(3, 192, 11, stride=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(192, 512, 5),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(512, 768, 3),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),

            nn.Conv2d(768, 768, 3),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),

            nn.Conv2d(768, 512, 3),
            nn.BatchNorm2d(512),
        )
        self.conv_r1 = nn.Conv2d(feat_in, feature_out*4*anchor, 3)
        self.conv_r2 = nn.Conv2d(feat_in, feature_out, 3)
        self.conv_cls1 = nn.Conv2d(feat_in, feature_out*2*anchor, 3)
        self.conv_cls2 = nn.Conv2d(feat_in, feature_out, 3)
        self.regress_adjust = nn.Conv2d(4*anchor, 4*anchor, 1)

        self.r1_kernel = []
        self.cls1_kernel = []

    def forward(self, x):
        x_f = self.featureExtract(x)
        # print(x_f)
        # clsout = F.conv2d(self.conv_cls2(x_f), self.cls1_kernel)
        # adjustres = self.regress_adjust(F.conv2d(self.conv_r2(x_f), self.r1_kernel))
        # weight = self.regress_adjust.weight.cpu().detach().numpy()
        # bias = self.regress_adjust.bias.cpu().detach().numpy()
        return self.regress_adjust(F.conv2d(self.conv_r2(x_f), self.r1_kernel)), F.conv2d(self.conv_cls2(x_f), self.cls1_kernel)

    def temple(self, z):
        print z.cpu().numpy(), "numpy z", z.shape
        znumpy = z.cpu().numpy().reshape(z.shape[0]*z.shape[1]*z.shape[2]*z.shape[3])
        z_f = self.featureExtract(z)
        # print z_f, " feature_out ", z_f.shape
        r1_kernel_raw = self.conv_r1(z_f)
        cls1_kernel_raw = self.conv_cls1(z_f)
        kernel_size = r1_kernel_raw.data.size()[-1]
        print r1_kernel_raw
        self.r1_kernel = r1_kernel_raw.view(self.anchor*4, self.feature_out, kernel_size, kernel_size)
        self.cls1_kernel = cls1_kernel_raw.view(self.anchor*2, self.feature_out, kernel_size, kernel_size)

class SiamRPNBG_Caffe2:
    def __init__(self):
        self.Init_Net = None
        self.Temple_Net = None
        self.Track_Net = None
        self.Net_Enum = ['init','temple','track','adjust','CorrelationFilter']

        curinput = 'data'
        for net in self.Net_Enum:
            model = model_helper.ModelHelper(name=net,arg_scope=cnn_arg_scope,)
            if net == 'init' or net == 'temple' or net == 'track':
                p = brew.conv(model,'data','conv_1',3,192,kernel=11,stride=2);
                p = brew.spatial_bn(model,'conv_1','bn_1',192,is_test=True,epsilon=1e-5,momentum=0.1);
                p = brew.relu(model,'bn_1','bn_1');
                p = brew.max_pool(model,'bn_1','pool_1',kernel=3,stride=2);

                p = brew.conv(model, 'pool_1', 'conv_2', 192, 512, kernel=5);
                p = brew.spatial_bn(model, 'conv_2', 'bn_2', 512,is_test=True,epsilon=1e-5,momentum=0.1);
                p = brew.relu(model, 'bn_2', 'bn_2');
                p = brew.max_pool(model, 'bn_2', 'pool_2', kernel=3, stride=2);

                p = brew.conv(model, 'pool_2', 'conv_3', 512, 768, kernel=3);
                p = brew.spatial_bn(model, 'conv_3', 'bn_3', 768,is_test=True,epsilon=1e-5,momentum=0.1);
                p = brew.relu(model, 'bn_3', 'bn_3');

                p = brew.conv(model, 'bn_3', 'conv_4', 768, 768, kernel=3);
                p = brew.spatial_bn(model, 'conv_4', 'bn_4', 768, is_test=True,epsilon=1e-5,momentum=0.1);
                p = brew.relu(model, 'bn_4', 'bn_4');

                p = brew.conv(model, 'bn_4', 'conv_5', 768, 512, kernel=3);
                p = brew.spatial_bn(model, 'conv_5', 'feature_out', 512, is_test=True,epsilon=1e-5,momentum=0.1);



                #rpn sub networks
                if net == 'init':
                    p = brew.conv(model,'feature_out','conv_r1',512,512*4*5,kernel=3)
                    p = brew.conv(model,'feature_out','conv_r2',512,512,kernel=3)
                    p = brew.conv(model,'feature_out','conv_cls1',512,512*2*5,kernel=3)
                    p = brew.conv(model,'feature_out','conv_cls2',512,512,kernel=3)
                    self.add_inference_inputs(model)
                    self.Init_Net = model.param_init_net
                    self.InitModel = model
                elif net == 'temple':
                    p = brew.conv(model, 'feature_out', 'conv_r1', 512, 512 * 4 * 5, kernel=3)
                    p = brew.conv(model, 'feature_out', 'conv_cls1', 512, 512 * 2 * 5, kernel=3)
                    self.add_inference_inputs(model)
                    self.Temple_Net = model.net
                    self.TempleModel = model
                elif net == 'track':
                    p = brew.conv(model, 'feature_out', 'conv_r2', 512, 512, kernel=3)
                    p = brew.conv(model, 'feature_out', 'conv_cls2', 512, 512, kernel=3)
                    self.add_inference_inputs(model)
                    self.Track_Net = model.net
                    self.TrackModel = model
            elif net == 'adjust':
                p = brew.conv(model, 'conv_conv_r2', 'r2_out', 4*5, 5*4, kernel=1)
                self.add_inference_inputs(model)
                self.addJustModel = model
            elif net == 'CorrelationFilter':
                p = brew.conv(model, 'conv_r2', 'conv_conv_r2', 4 * 5, 5 * 4, kernel=1)
                self.add_inference_inputs(model)
                self.CorrelationFilterModel = model


    def add_inference_inputs(self,model):
        """Create network input blobs used for inference."""

        def create_input_blobs_for_net(net_def):
            for op in net_def.op:
                for blob_in in op.input:
                    if not workspace.HasBlob(blob_in):
                        workspace.CreateBlob(blob_in)

        create_input_blobs_for_net(model.net.Proto())

    def load_weights_from_torch_model(self):
        pass



core.GlobalInit(['caffe2', '--caffe2_log_level=0'])
workspace.ResetWorkspace("/opt")
DaSiam=SiamRPNBG_Caffe2()
workspace.RunNetOnce(DaSiam.Init_Net)
workspace.RunNetOnce(DaSiam.addJustModel.param_init_net)
workspace.RunNetOnce(DaSiam.CorrelationFilterModel.param_init_net)
img = cv2.imread('/opt/out.jpg')
b, g, r = cv2.split(img)
rgb_img = cv2.merge([r, g, b])
sized = cv2.resize(rgb_img, (255, 255), interpolation=cv2.INTER_CUBIC)
# exportaspb.export(workspace=workspace,net=DaSiam.InitModel.net,params=DaSiam.InitModel.params,init_net_name="/opt/global_init_net.pb",predict_net_name="/opt/global_pred_net.pb")
npar = np.array(sized)
pp = np.ascontiguousarray(np.transpose(npar,[2,0,1])).reshape(1,3,sized.shape[0],sized.shape[1]).astype(np.float32)/255.0
# print(DaSiam.Temple_Net.Proto())
workspace.CreateNet(DaSiam.Temple_Net)
workspace.CreateNet(DaSiam.Track_Net)

workspace.FeedBlob('data', pp)
workspace.RunNet(DaSiam.Temple_Net)
# workspace.RunNet(DaSiam.Track_Net)


pnet = SiamRPNBIG()

pnet.load_state_dict(torch.load('/opt/about_tracking/SiamRPNBIG.model'))

# print(pnet(torch.Tensor(pp)))
conv_idx = 0
bn_idx = 0
pytorch_opindex_to_caffe2_op_input0 = {
    1:'conv_1_w',
    2:'conv_2_w',
    3:'conv_3_w',
    4:'conv_4_w',
    5:'conv_5_w',
    6:'conv_r1_w',
    7:'conv_r2_w',
    8:'conv_cls1_w',
    9:'conv_cls2_w',
    10:'r2_out_w'
}

pytorch_bnopindex_to_caffe2_bnop_input0 = {
    1:'conv_1',
    2:'conv_2',
    3:'conv_3',
    4:'conv_4',
    5:'conv_5',
}
# print(DaSiam.Temple_Net.Proto())
def get_conv_op_by_input0(Net,tNet,aNet,input):
    for op in Net.Proto().op:
        if op.type == 'Conv' and op.input[1] == input:
            return op
    for op in tNet.Proto().op:
        if op.type == 'Conv' and op.input[1] == input:
            return op
    for op in aNet.Proto().op:
        if op.type == 'Conv' and op.input[1] == input:
            return op
    raise RuntimeError

def get_bn_op_by_input0(Net,tNet,input):
    for op in Net.Proto().op:
        if op.type == 'SpatialBN' and op.input[0] == input:
            return op
    for op in tNet.Proto().op:
        if op.type == 'SpatialBN' and op.input[0] == input:
            return op
    raise RuntimeError

for m in enumerate(pnet.modules()):
    if isinstance(m[1],torch.nn.Conv2d):
        conv_idx +=1
        # if conv_idx == 10:
        #     continue
        weight = m[1].weight.detach().numpy()
        bias   = m[1].bias.detach().numpy()
        op = get_conv_op_by_input0(DaSiam.Temple_Net,DaSiam.Track_Net,DaSiam.addJustModel,
                                   pytorch_opindex_to_caffe2_op_input0[conv_idx])
        # print(weight)
        assert op
        ws_blob = workspace.FetchBlob(op.input[1])
        # print(op.input[1])
        workspace.FeedBlob(op.input[1],weight.reshape(ws_blob.shape).astype(np.float32, copy=False))
        ws_blob = workspace.FetchBlob(op.input[2])
        workspace.FeedBlob(op.input[2], bias.reshape(ws_blob.shape).astype(np.float32, copy=False))
        # print(ws_blob)

    if isinstance(m[1],torch.nn.BatchNorm2d):
        bn_idx +=1
        weight = m[1].weight.detach().numpy()
        bias   = m[1].bias.detach().numpy()
        running_mean = m[1].running_mean.detach().numpy()
        running_var  = m[1].running_var.detach().numpy()
        # print(weight)
        # print(bias)
        # print(running_mean)
        # print(running_var)
        op = get_bn_op_by_input0(DaSiam.Temple_Net,DaSiam.Track_Net,pytorch_bnopindex_to_caffe2_bnop_input0[bn_idx])

        assert op
        ws_blob = workspace.FetchBlob(op.input[1])
        workspace.FeedBlob(op.input[1], weight.reshape(ws_blob.shape).astype(np.float32, copy=False))

        ws_blob = workspace.FetchBlob(op.input[2])
        workspace.FeedBlob(op.input[2], bias.reshape(ws_blob.shape).astype(np.float32, copy=False))

        ws_blob = workspace.FetchBlob(op.input[3])
        workspace.FeedBlob(op.input[3], running_mean.reshape(ws_blob.shape).astype(np.float32, copy=False))

        ws_blob = workspace.FetchBlob(op.input[4])
        workspace.FeedBlob(op.input[4], running_var.reshape(ws_blob.shape).astype(np.float32, copy=False))

import os
#FOR CHECK CAFFE2 MODEL IS EXPORT CORRECT
workspace.FeedBlob('data', pp)
workspace.RunNet(DaSiam.Temple_Net)
workspace.RunNet(DaSiam.Track_Net)

c2_out = workspace.FetchBlob("conv_r1")
print("c2_r1:")
print(c2_out)

c2_out = workspace.FetchBlob("conv_cls1")
print("c2_cls1:")
print(c2_out)

z = torch.Tensor(pp)
pip = None
pnet.train(False)
pnet.temple(z)
#CHECK END





def dump_proto_files(model, output_dir,initfile_name,predictfile_name):
    """Save prototxt descriptions of the training network and parameter
    initialization network."""
    with open(os.path.join(output_dir, predictfile_name), 'w') as fid:
        fid.write(model.net.Proto().SerializeToString())
    with open(os.path.join(output_dir, initfile_name), 'w') as fid:
        fid.write(str(model.param_init_net.Proto().SerializeToString()))

def savemodelfile(model):
    params = []
    for i in range(1,4):
        params.append("bn_{}_riv".format(i))
        params.append("bn_{}_rm".format(i))
    params.append("feature_out_rm")
    params.append("feature_out_riv")
    for b in model.params:
        params.append(str(b))
    pe_meta = PredictorExportMeta(predict_net=model.net.Proto(),
                                  parameters=params,
                                  inputs=["data"],
                                  outputs=["conv_r1","conv_cls1"]
                                  )
    save_to_db("minidb", "/opt/DasiamRPN.minidb", pe_meta)
    print ("The deploy model is saved to: /opt/DasimRPN.minidb")

savemodelfile(DaSiam.InitModel)
dump_proto_files(DaSiam.TrackModel,"/opt","track_init.pbtxt","track_pred.pbtxt")
dump_proto_files(DaSiam.TempleModel,"/opt","temple_init.pbtxt","temple_pred.pbtxt")
dump_proto_files(DaSiam.InitModel,"/opt","global_init.pbtxt","global_pred.pbtxt")


exportaspb.export(workspace=workspace,net=DaSiam.InitModel.net,params=DaSiam.InitModel.params,init_net_name="/opt/global_init_net.pb",predict_net_name="/opt/global_pred_net.pb")
exportaspb.export(workspace=workspace,net=DaSiam.TempleModel.net,params=DaSiam.TempleModel.params,init_net_name="/opt/temple_init_net.pb",predict_net_name="/opt/temple_pred_net.pb")
exportaspb.export(workspace=workspace,net=DaSiam.TrackModel.net,params=DaSiam.TrackModel.params,init_net_name="/opt/track_init_net.pb",predict_net_name="/opt/track_pred_net.pb")

exportaspb.export(workspace=workspace,net=DaSiam.addJustModel.net,params=DaSiam.addJustModel.params,init_net_name="/opt/adjust_init_net.pb",predict_net_name="/opt/adjust_pred_net.pb")

exportaspb.export(workspace=workspace,net=DaSiam.CorrelationFilterModel.net,params=DaSiam.CorrelationFilterModel.params,init_net_name="/opt/Correlation_init_net.pb",predict_net_name="/opt/Correlation_pred_net.pb")

