import my_resnet
from my_resnet import Bottleneck
import torch
import torchvision
import os


import torch
from codes import mvtecad
from functools import reduce
from torch.utils.data import DataLoader
from codes.datasets import *
from codes.networks import *
from codes.networks import multi_center_res_backbone
from codes.inspection import eval_encoder_NN_multiK
from codes.utils import *
from datasets import Multi_Center_Dataset_res


import numpy as np
import cv2
def cnn_output_size(H, K, S=1, P=0) -> int:
    """

    :param int H: input_size
    :param int K: filter_size
    :param int S: stride
    :param int P: padding
    :return:
    """
    return 1 + (H - K + 2 * P) // S

def crop_CHW(image, i, j, K, S=1):
    if S == 1:
        h, w = i, j
    else:
        h = S * i
        w = S * j
    return image[:, h: h + K, w: w + K]


def load_pre_train_model(pretrained,model):
    pretrained_dict = pretrained

    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                       if k in model_dict.keys()}
    # for k, _ in pretrained_dict.items():
    #    logger.info(
    #        '=> loading {} pretrained model {}'.format(k, pretrained))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
def save_model(net, epoch):
    """Save the model at "checkpoint_interval" and its multiple"""
    model_name = epoch+".pth"
    model_name = os.path.join("/media/ubuntu/Backup/data/MTV_benchmark/my_zipper/model/", model_name)
    #model output path
    torch.save(net.state_dict(), model_name)



enc = my_resnet.ResNet(Bottleneck,[3, 4, 6, 3],1000).cuda()

ckpt = torch.load("./resnet50-19c8e357.pth")

model_dict = enc.state_dict()
load_pre_train_model(ckpt,enc)

# img = torch.randn(1,3,256,256).cuda()
# out = enc(img)
# aaa =1


D = 1000
lr = 0.0001
lambda_value = 1
rep =100
train_x = mvtecad.get_x_standardized("my_zipper", mode='train')
train_x = NHWC2NCHW(train_x)
datasets = dict()


datasets[f'msvdd_64'] = Multi_Center_Dataset_res(train_x, K=256, repeat=rep)


dataset = DictionaryConcatDataset(datasets)
loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)

#enc.load("my_zipper")


multi_cls = multi_center_res_backbone(D,20).cuda()
#multi_cls.load("tt")
modules = [enc,multi_cls]
params = [list(module.parameters()) for module in modules]
params = reduce(lambda x, y: x + y, params)
opt = torch.optim.Adam(params=params, lr=lr)
for i in range(100):

    for d in loader:
        # for i in range(5):
        #
        #     d = to_device(d, 'cuda', non_blocking=True)
        #     opt.zero_grad()
        #
        #     loss_pos_64 = PositionClassifier.infer(cls_64, enc, d['pos_64'])
        #     loss_pos_32 = PositionClassifier.infer(cls_32, enc.enc, d['pos_32'])
        #     loss_svdd_64 = Multi_Center_Dataset.infer(enc,multi_cls, d['msvdd_64'])
        #     loss_svdd_32 = Multi_Center_Dataset.infer(enc.enc,multi_cls, d['msvdd_64'])
        #
        #     loss = (loss_pos_64 + loss_pos_32)
        #     print((loss_pos_64 + loss_pos_32).item(),"    ",(loss_svdd_32+loss_svdd_64).item())
        #
        #     loss.backward()
        #     opt.step()
        #     print(loss.item())
        d = to_device(d, 'cuda', non_blocking=True)
        opt.zero_grad()


        loss_svdd_64 = Multi_Center_Dataset_res.infer(enc, multi_cls, d['msvdd_64'])


        loss = 0.0001 * loss_svdd_64
        print(i, "     ", loss_svdd_64.item())

        loss.backward()
        opt.step()
        print(loss.item())

save_model(enc,"enc")
save_model(multi_cls,"mul")