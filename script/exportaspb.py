# DaSiamRPNCaffe2
# Licensed under The MIT License
# Written by wzq
# Reference:Detectron
# -------------------------------------------------------
from caffe2.python import core, model_helper, net_drawer, workspace, visualize, brew
from caffe2.python.predictor.mobile_exporter import Export


def export(workspace, net, params, init_net_name="/opt/DasimRPN_init_net.pb", predict_net_name="/opt/DasiamRPN_predict_net.pb" ):
    extra_params = []
    extra_blobs = []
    for blob in workspace.Blobs():
        name = str(blob)
        if name.endswith("_rm") or name.endswith("_riv"):
            extra_params.append(name)
            extra_blobs.append(workspace.FetchBlob(name))
    for name, blob in zip(extra_params, extra_blobs):
        workspace.FeedBlob(name, blob)
        params.append(name)
    for p in params:
        print p
    init_net, predict_net = Export(workspace, net, params)
    with open(init_net_name, 'wb') as f:
        f.write(init_net.SerializeToString())
    with open(predict_net_name, 'wb') as f:
        f.write(predict_net.SerializeToString())
