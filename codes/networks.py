import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import os
import numpy as np
import cv2
def makedirpath(fpath: str):
    dpath = os.path.dirname(fpath)
    if dpath:
        os.makedirs(dpath, exist_ok=True)

__all__ = ['EncoderHier', 'Encoder', 'PositionClassifier']


class Encoder(nn.Module):
    def __init__(self, K, D=64, bias=True):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 5, 2, 0, bias=bias)
        self.conv2 = nn.Conv2d(64, 64, 5, 2, 0, bias=bias)
        self.conv3 = nn.Conv2d(64, 128, 5, 2, 0, bias=bias)
        self.conv4 = nn.Conv2d(128, D, 5, 1, 0, bias=bias)

        self.K = K
        self.D = D
        self.bias = bias

    def forward(self, x):
        h = self.conv1(x)
        h = F.leaky_relu(h, 0.1)

        h = self.conv2(h)
        h = F.leaky_relu(h, 0.1)

        h = self.conv3(h)

        if self.K == 64:
            h = F.leaky_relu(h, 0.1)
            h = self.conv4(h)

        h = torch.tanh(h)

        return h

    def save(self, name):
        fpath = self.fpath_from_name(name)
        makedirpath(fpath)
        torch.save(self.state_dict(), fpath)

    def load(self, name):
        fpath = self.fpath_from_name(name)
        self.load_state_dict(torch.load(fpath))

    @staticmethod
    def fpath_from_name(name):
        return f'ckpts/{name}/encoder_nohier.pkl'


def forward_hier(x, emb_small, K):
    K_2 = K // 2
    n = x.size(0)
    x1 = x[..., :K_2, :K_2]
    x2 = x[..., :K_2, K_2:]
    x3 = x[..., K_2:, :K_2]
    x4 = x[..., K_2:, K_2:]
    xx = torch.cat([x1, x2, x3, x4], dim=0)
    hh = emb_small(xx)

    h1 = hh[:n]
    h2 = hh[n: 2 * n]
    h3 = hh[2 * n: 3 * n]
    h4 = hh[3 * n:]

    h12 = torch.cat([h1, h2], dim=3)
    h34 = torch.cat([h3, h4], dim=3)
    h = torch.cat([h12, h34], dim=2)
    return h


class EncoderDeep(nn.Module):
    def __init__(self, K, D=64, bias=True):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=bias)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 0, bias=bias)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 0, bias=bias)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 0, bias=bias)
        self.conv5 = nn.Conv2d(128, 64, 3, 1, 0, bias=bias)
        self.conv6 = nn.Conv2d(64, 32, 3, 1, 0, bias=bias)
        self.conv7 = nn.Conv2d(32, 32, 3, 1, 0, bias=bias)
        self.conv8 = nn.Conv2d(32, D, 3, 1, 0, bias=bias)

        self.K = K
        self.D = D

    def forward(self, x):
        h = self.conv1(x)
        h = F.leaky_relu(h, 0.1)

        h = self.conv2(h)
        h = F.leaky_relu(h, 0.1)

        h = self.conv3(h)
        h = F.leaky_relu(h, 0.1)

        h = self.conv4(h)
        h = F.leaky_relu(h, 0.1)

        h = self.conv5(h)
        h = F.leaky_relu(h, 0.1)

        h = self.conv6(h)
        h = F.leaky_relu(h, 0.1)

        h = self.conv7(h)
        h = F.leaky_relu(h, 0.1)

        h = self.conv8(h)
        h = torch.tanh(h)

        return h

    def save(self, name):
        fpath = self.fpath_from_name(name)
        makedirpath(fpath)
        torch.save(self.state_dict(), fpath)

    def load(self, name):
        fpath = self.fpath_from_name(name)
        self.load_state_dict(torch.load(fpath))

    @staticmethod
    def fpath_from_name(name):
        return f'ckpts/{name}/encdeep.pkl'


class EncoderHier(nn.Module):
    def __init__(self, K, D=64, bias=True):
        super().__init__()

        if K > 64:
            self.enc = EncoderHier(K // 2, D, bias=bias)

        elif K == 64:
            self.enc = EncoderDeep(K // 2, D, bias=bias)

        else:
            raise ValueError()

        self.conv1 = nn.Conv2d(D, 128, 2, 1, 0, bias=bias)
        self.conv2 = nn.Conv2d(128, D, 1, 1, 0, bias=bias)

        self.K = K
        self.D = D

    def forward(self, x):
        h = forward_hier(x, self.enc, K=self.K)

        h = self.conv1(h)
        h = F.leaky_relu(h, 0.1)

        h = self.conv2(h)
        h = torch.tanh(h)

        return h

    def save(self, name):
        fpath = self.fpath_from_name(name)
        makedirpath(fpath)
        torch.save(self.state_dict(), fpath)

    def load(self, name):
        fpath = self.fpath_from_name(name)
        self.load_state_dict(torch.load(fpath))

    @staticmethod
    def fpath_from_name(name):
        return f'ckpts/{name}/enchier.pkl'


################


xent = nn.CrossEntropyLoss()


class NormalizedLinear(nn.Module):
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(NormalizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        with torch.no_grad():
            w = self.weight / self.weight.data.norm(keepdim=True, dim=0)
        return F.linear(x, w, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class PositionClassifier(nn.Module):
    def __init__(self, K, D, class_num=8):
        super().__init__()
        self.D = D

        self.fc1 = nn.Linear(D, 128)
        self.act1 = nn.LeakyReLU(0.1)

        self.fc2 = nn.Linear(128, 128)
        self.act2 = nn.LeakyReLU(0.1)

        self.fc3 = NormalizedLinear(128, class_num)

        self.K = K

    def save(self, name):
        fpath = self.fpath_from_name(name)
        makedirpath(fpath)
        torch.save(self.state_dict(), fpath)

    def load(self, name):
        fpath = self.fpath_from_name(name)
        self.load_state_dict(torch.load(fpath))

    def fpath_from_name(self, name):
        return f'ckpts/{name}/position_classifier_K{self.K}.pkl'

    @staticmethod
    def infer(c, enc, batch):
        x1s, x2s, ys = batch

        h1 = enc(x1s)
        h2 = enc(x2s)

        logits = c(h1, h2)
        loss = xent(logits, ys)
        return loss

    def forward(self, h1, h2):
        h1 = h1.view(-1, self.D)
        h2 = h2.view(-1, self.D)

        h = h1 - h2

        h = self.fc1(h)
        h = self.act1(h)

        h = self.fc2(h)
        h = self.act2(h)

        h = self.fc3(h)
        return h


class multi_center(nn.Module):
    def __init__(self, inc, center_num):
        super().__init__()
        self.D = inc
        #self.centers = nn.Conv1d(inc, center_num, 1)

        self.myfc1 = nn.Linear(inc, inc*4)
        self.myfc = nn.Linear(inc, center_num)
        self.centers = nn.Parameter(torch.zeros(inc * 4, center_num))

        self.pooling = nn.AdaptiveAvgPool2d((1,1))
        self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion = torch.nn.MSELoss()
        nn.init.normal(self.myfc1.weight, std=0.01)
        self.nm = nn.InstanceNorm1d(1)

        nn.init.normal(self.myfc.weight,std=0.1)

    def forward(self, h1):
        b,c,w,h = h1.size()
        #img_py = h1.clone().data.permute(0, 1, 2,3)[0,: :, :].cpu().numpy()

        # img_py[img_py < 0] = 0
        # print(np.max(img_py))
        # print(np.min(img_py))
        # img_py = np.sum(img_py, axis=2)
        # img_py = img_py / np.max(img_py)
        #
        # img_py = img_py.astype(np.float)
        #
        # cv2.namedWindow("4", 0)
        # cv2.imshow("4", img_py)
        #
        # cv2.waitKey(0)
        h1 = self.pooling(h1)
        h1 = torch.flatten(h1, 1).unsqueeze(1)
        h1 = self.nm(h1)[:,0,:]
        #h1 = h1.permute(0,2,1)
        #h1 = self.centers(h1).view(b,-1)
        #h1 = self.myfc1(h1)

        h1 = self.myfc(h1)
        myfc_w = self.myfc.weight.permute(1,0)


        out = torch.argmax(h1, dim=1)
        center = myfc_w[out[0],:]

        print(out)
        #out_tensor = out
        loss = self.criterion(h1,center)
        #print("lll   ",loss)
        return loss
    def fpath_from_name(self, name):
        return f'ckpts/{name}/multi_cls.pkl'
    def save(self, name):
        fpath = self.fpath_from_name(name)
        makedirpath(fpath)
        torch.save(self.state_dict(), fpath)

    def load(self, name):
        fpath = self.fpath_from_name(name)
        self.load_state_dict(torch.load(fpath))








class multi_center(nn.Module):
    def __init__(self, inc, center_num):
        super().__init__()
        self.D = inc
        #self.centers = nn.Conv1d(inc, center_num, 1)

        self.myfc1 = nn.Linear(inc, inc*4)
        self.myfc = nn.Linear(inc, center_num)
        self.centers = nn.Parameter(torch.zeros(inc * 4, center_num))

        self.pooling = nn.AdaptiveAvgPool2d((1,1))
        self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion = torch.nn.MSELoss()
        nn.init.normal(self.myfc1.weight, std=0.01)
        self.nm = nn.InstanceNorm1d(1)

        nn.init.normal(self.myfc.weight,std=0.1)

    def forward(self, h1):
        b,c,w,h = h1.size()
        #img_py = h1.clone().data.permute(0, 1, 2,3)[0,: :, :].cpu().numpy()

        # img_py[img_py < 0] = 0
        # print(np.max(img_py))
        # print(np.min(img_py))
        # img_py = np.sum(img_py, axis=2)
        # img_py = img_py / np.max(img_py)
        #
        # img_py = img_py.astype(np.float)
        #
        # cv2.namedWindow("4", 0)
        # cv2.imshow("4", img_py)
        #
        # cv2.waitKey(0)
        h1 = self.pooling(h1)
        h1 = torch.flatten(h1, 1).unsqueeze(1)
        h1 = self.nm(h1)[:,0,:]
        #h1 = h1.permute(0,2,1)
        #h1 = self.centers(h1).view(b,-1)
        #h1 = self.myfc1(h1)

        h1 = self.myfc(h1)
        myfc_w = self.myfc.weight.permute(1,0)


        out = torch.argmax(h1, dim=1)
        center = myfc_w[out[0],:]

        print(out)
        #out_tensor = out
        loss = self.criterion(h1,center)
        #print("lll   ",loss)
        return loss
    def fpath_from_name(self, name):
        return f'ckpts/{name}/multi_cls.pkl'
    def save(self, name):
        fpath = self.fpath_from_name(name)
        makedirpath(fpath)
        torch.save(self.state_dict(), fpath)

    def load(self, name):
        fpath = self.fpath_from_name(name)
        self.load_state_dict(torch.load(fpath))






class multi_center_res_backbone(nn.Module):
    def __init__(self, inc, center_num):
        super().__init__()
        self.D = inc
        #self.centers = nn.Conv1d(inc, center_num, 1)

        self.myfc1 = nn.Linear(inc, inc*4)
        self.myfc = nn.Linear(inc, center_num)
        self.centers = nn.Parameter(torch.zeros(inc * 4, center_num))

        self.pooling = nn.AdaptiveAvgPool2d((1,1))
        self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion = torch.nn.MSELoss()
        nn.init.normal(self.myfc1.weight, std=0.01)
        self.nm = nn.InstanceNorm1d(1)

        nn.init.normal(self.myfc.weight,std=0.1)

    def forward(self, h1):
        b,c = h1.size()
        #img_py = h1.clone().data.permute(0, 1, 2,3)[0,: :, :].cpu().numpy()

        # img_py[img_py < 0] = 0
        # print(np.max(img_py))
        # print(np.min(img_py))
        # img_py = np.sum(img_py, axis=2)
        # img_py = img_py / np.max(img_py)
        #
        # img_py = img_py.astype(np.float)
        #
        # cv2.namedWindow("4", 0)
        # cv2.imshow("4", img_py)
        #
        # cv2.waitKey(0)
        #h1 = self.pooling(h1)
        h1 = torch.flatten(h1, 1).unsqueeze(1)
        h1 = self.nm(h1)[:,0,:]
        #h1 = h1.permute(0,2,1)
        #h1 = self.centers(h1).view(b,-1)
        #h1 = self.myfc1(h1)

        h1 = self.myfc(h1)
        myfc_w = self.myfc.weight.permute(1,0)


        out = torch.argmax(h1, dim=1)
        center = myfc_w[out[0],:]

        print(out)
        #out_tensor = out
        loss = self.criterion(h1,center)
        #print("lll   ",loss)
        return loss
    def fpath_from_name(self, name):
        return f'ckpts/{name}/multi_cls.pkl'
    def save(self, name):
        fpath = self.fpath_from_name(name)
        makedirpath(fpath)
        torch.save(self.state_dict(), fpath)

    def load(self, name):
        fpath = self.fpath_from_name(name)
        self.load_state_dict(torch.load(fpath))




class multi_center_res_backbone_test(nn.Module):
    def __init__(self, inc, center_num):
        super().__init__()
        self.D = inc
        #self.centers = nn.Conv1d(inc, center_num, 1)

        self.myfc1 = nn.Linear(inc, inc*4)
        self.myfc = nn.Linear(inc, center_num)
        self.centers = nn.Parameter(torch.zeros(inc * 4, center_num))

        self.pooling = nn.AdaptiveAvgPool2d((1,1))
        self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion = torch.nn.MSELoss()
        nn.init.normal(self.myfc1.weight, std=0.01)
        self.nm = nn.InstanceNorm1d(1)

        #nn.init.normal(self.myfc.weight,std=0.1)

    def forward(self, h1):
        b,c = h1.size()

        h1 = torch.flatten(h1, 1).unsqueeze(1)
        h1 = self.nm(h1)[:,0,:]
        #h1 = h1.permute(0,2,1)
        #h1 = self.centers(h1).view(b,-1)
        #h1 = self.myfc1(h1)

        h1 = self.myfc(h1)
        myfc_w = self.myfc.weight.permute(1,0)


        out = torch.argmax(h1, dim=1)
        center = myfc_w[out[0],:]

        print(out)
        #out_tensor = out
        loss = self.criterion(h1,center)
        #print("lll   ",loss)
        return out,loss
    def fpath_from_name(self, name):
        return f'ckpts/{name}/multi_cls.pkl'
    def save(self, name):
        fpath = self.fpath_from_name(name)
        makedirpath(fpath)
        torch.save(self.state_dict(), fpath)

    def load(self, name):
        fpath = self.fpath_from_name(name)
        self.load_state_dict(torch.load(fpath))





class multi_center_test(nn.Module):
    def __init__(self, inc, center_num):
        super().__init__()
        self.D = inc
        # self.centers = nn.Conv1d(inc, center_num, 1)

        self.myfc1 = nn.Linear(inc, inc * 4)
        self.myfc = nn.Linear(inc, center_num)
        self.centers = nn.Parameter(torch.zeros(inc * 4, center_num))


        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion = torch.nn.MSELoss()
        nn.init.normal(self.myfc1.weight, std=5)
        self.nm = nn.InstanceNorm1d(1)

        nn.init.normal(self.myfc.weight, std=0.1)

    def forward(self, h1):
        b, c, w, h = h1.size()
        h1 = self.pooling(h1)
        h1 = torch.flatten(h1, 1).unsqueeze(1)
        h1 = self.nm(h1)[:, 0, :]
        # h1 = h1.permute(0,2,1)
        # h1 = self.centers(h1).view(b,-1)
        # h1 = self.myfc1(h1)

        #h1 = self.myfc1(h1)

        h1 = self.myfc(h1)
        myfc_w = self.myfc.weight.permute(1, 0)

        out = torch.argmax(h1, dim=1)
        center = myfc_w[out[0], :]

        #print(out)
        # out_tensor = out
        loss = self.criterion(h1, center)
        # print("lll   ",loss)
        return out,loss

    def fpath_from_name(self, name):
        return f'ckpts/{name}/multi_cls.pkl'

    def save(self, name):
        fpath = self.fpath_from_name(name)
        makedirpath(fpath)
        torch.save(self.state_dict(), fpath)

    def load(self, name):
        fpath = self.fpath_from_name(name)
        self.load_state_dict(torch.load(fpath))


if __name__ == "__main__":




    test_model = multi_center(256,10)
    img = torch.randn(1, 256, 10, 10)
    out = test_model(img)
    print(out)




