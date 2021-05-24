import torch
from codes.networks import multi_center_res_backbone_test
import numpy as np
import cv2
import my_resnet
from my_resnet import Bottleneck
import glob
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




enc = my_resnet.ResNet(Bottleneck,[3, 4, 6, 3],1000).cuda()

ckpt = torch.load("./model/enc.pth")

enc.load_state_dict(ckpt)




def img_infer(img_name):
    a = np.zeros(20)
    img = cv2.imread(img_name)
    mean = np.mean(img)
    img = (img - mean) / 255
    img = cv2.resize(img,(1024,1024))
    # cv2.imshow("11",img)
    # cv2.waitKey(0)
    row = cnn_output_size(1024,256,48)
    col = cnn_output_size(1024,256,48)
    img = np.transpose(img, [2, 0, 1])

    for i in range(row):
        for j in range(col):

            img_p = crop_CHW(img, i, j, 256, 48)
            aa = np.transpose(img_p, [1, 2, 0])
            in_tensor = torch.from_numpy(img_p.astype(np.float32)).contiguous()
            in_tensor = in_tensor.unsqueeze(0).cuda()
            out = enc(in_tensor)
            out, dis = multi_cls(out)

            print(out, "   ", dis)
            # out_f = torch.nn.functional.softmax(out_f, dim=1)
            if a[out[0]] < dis:
                a[out[0]] = dis
                print(a)
            cv2.namedWindow("p",0)
            cv2.imshow("p", aa)


            cv2.waitKey(0)
            # if dis >0.72 and out[0]==3:
            #     aa =cv2.resize(aa,(256,256))
            #     cv2.imshow("p",aa)
            #     cv2.waitKey(0)
            # print(torch.argmax(out_f,dim=1))
    return a



if __name__ == "__main__":
    D = 1000
    multi_cls = multi_center_res_backbone_test(D, 20).cuda()
    multi_cls.load_state_dict(torch.load("./model/mul.pth"))
    enc.eval()
    multi_cls.eval()

    ok_center_dist = np.zeros(20)

    for img_name in glob.glob("./ok/*.jpg"):
        img = cv2.imread(img_name)
        mean = np.mean(img)
        img = (img - mean) / 255
        img = cv2.resize(img, (1024, 1024))
        # cv2.imshow("11",img)
        # cv2.waitKey(0)
        row = cnn_output_size(1024, 256, 48)
        col = cnn_output_size(1024, 256, 48)
        img = np.transpose(img, [2, 0, 1])

        for i in range(row):
            for j in range(col):


                img_p = crop_CHW(img,i,j,256,48)
                aa = np.transpose(img_p,[ 1, 2, 0])
                in_tensor = torch.from_numpy(img_p.astype(np.float32)).contiguous()
                in_tensor = in_tensor.unsqueeze(0).cuda()
                out = enc(in_tensor)
                out,dis = multi_cls(out)

                print(out,"   ",dis)
    print(ok_center_dist)



