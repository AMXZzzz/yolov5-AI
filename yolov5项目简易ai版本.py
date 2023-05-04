# --*- coding: utf-8 -*--
# @Name : bilibili:随风而息
# @Time : 2023/5/4 11:08
# -----------------------
import argparse
import dxcam        # pip install dxcam
import cv2          # 最高别超过4.5.5.64 版本
import numpy as np
import torch
import win32api     # pip install pywin32
import win32con
from torch import nn

from models.experimental import Ensemble
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes, xyxy2xywh

parser = argparse.ArgumentParser()
parser.add_argument("--module-path", "-m", type=str, default="../weights/cf_v5s_3w1_2l.pt", help="模型路径")
parser.add_argument("--image-size", type=int, default=640, help="模型输入大小")
parser.add_argument("--conf-thres", type=float, default=0.35, help="置信度")
parser.add_argument("--iou-thres", type=float, default=0.03, help="iou")
parser.add_argument("--device", type=str, default="", help="运行设备,为空自动选择,parame: cpu, 0, 1, ...")
parser.add_argument("--half", type=bool, default=True, help="半精度")

args = parser.parse_args()


class Yolov5:
    """
    Aim项目推理部分封装
    """
    def __init__(self, half, size):
        self.model = None
        self.stride = None
        self.device = ""
        self.half = half
        self.size = size

    def SetDevice(self, device):
        if device != "cpu":
            if torch.cuda.is_available():
                self.device = "cuda"

    def __GetModule(self, weights):
        # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
        from models.yolo import Detect, Model

        model = Ensemble()
        for w in weights if isinstance(weights, list) else [weights]:
            ckpt = torch.load(w, map_location='cpu')  # load
            ckpt = (ckpt.get('ema') or ckpt['model']).to(self.device).float()  # FP32 model

            # Model compatibility updates
            if not hasattr(ckpt, 'stride'):
                ckpt.stride = torch.tensor([32.])
            if hasattr(ckpt, 'names') and isinstance(ckpt.names, (list, tuple)):
                ckpt.names = dict(enumerate(ckpt.names))  # convert to dict

            model.append(
                ckpt.fuse().eval() if True and hasattr(ckpt, 'fuse') else ckpt.eval())  # model in eval mode

        # Module compatibility updates
        for m in model.modules():
            t = type(m)
            if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
                m.inplace = True  # torch 1.7.0 compatibility
                if t is Detect and not isinstance(m.anchor_grid, list):
                    delattr(m, 'anchor_grid')
                    setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
            elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
                m.recompute_scale_factor = None  # torch 1.11.0 compatibility

        # Return model
        if len(model) == 1:
            return model[-1]

        # Return detection ensemble
        print(f'Ensemble created with {weights}\n')
        for k in 'names', 'nc', 'yaml':
            setattr(model, k, getattr(model[0], k))
        model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
        assert all(
            model[0].nc == m.nc for m in model), f'Models have different class counts: {[m.nc for m in model]}'
        return model

    def LoadModel(self, weights):
        self.model = self.__GetModule(weights)
        if self.half:
            self.model.half()
        if self.device != "cpu":
            self.model(torch.zeros(1, 3, self.size, self.size).to(self.device).type_as(next(self.model.parameters())))
        self.stride = int(self.model.stride.max())

    def __Preproces(self, img):
        im = letterbox(img, self.size, stride=self.stride)[0]
        im = im.transpose((2, 0, 1))[::-1]
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.half else im.float()
        im /= 255.
        if len(im.shape) == 3:
            im = im[None]
        return im

    @staticmethod
    def __Postproces(pred, img, im):
        bbox_arr = []
        for i, det in enumerate(pred):
            gn = torch.tensor(img.shape)[[1, 0, 1, 0]]
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    x_c, x_y = (img.shape[1] * xywh[0]), (img.shape[0] * xywh[1])
                    width, height = img.shape[1] * xywh[2], img.shape[0] * xywh[3]
                    line = [x_c, x_y, width, height, float(conf), int(cls)]
                    bbox_arr.append(line)
        return bbox_arr # 返回的是所有目标的信息列表，[x,y,width,height,conf,class]

    def Inference(self, img, conf, iou):
        im = self.__Preproces(img)
        pred = self.model(im, augment=False, visualize=False)[0]
        pred = non_max_suppression(pred, conf_thres=conf, iou_thres=iou, agnostic=False, max_det=20)
        return self.__Postproces(pred, img, im)


def DarwBox(img, arr):
    for temp in arr:    # 遍历目标信息列表
        top_left = int(temp[0] - (temp[2] * 0.5)), int(temp[1] - (temp[3] * 0.5))
        bottom_right = int(temp[0] + (temp[2] * 0.5)), int(temp[1] + (temp[3] * 0.5))
        conf = temp[4]  # 目标置信度
        classes = temp[5]   # 目标的类别索引
        cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), thickness=2)
    cv2.imshow("test", img)
    cv2.waitKey(1)


class Capture:
    """
    dx截图函数封装
    """
    def __init__(self, size):
        self.dx = dxcam.create()
        self.size = size
        self.w = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
        self.h = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
        self.x0 = (self.w / 2) - (self.size / 2)
        self.y0 = (self.h / 2) - (self.size / 2)
        self.x1 = (self.w / 2) + (self.size / 2)
        self.y1 = (self.h / 2) + (self.size / 2)
        self.region = (int(self.x0), int(self.y0), int(self.x1), int(self.y1))

    def grab(self):
        img = self.dx.grab(self.region)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            return img
        return False


def run():
    # 设置截图范围
    cap = Capture(args.image_size)
    # 实列化
    yolo = Yolov5(args.half, args.image_size)
    # 设置运行设备
    yolo.SetDevice(args.device)
    # 加载pt模型
    yolo.LoadModel(args.module_path)
    # 循环
    while True:
        # 截图
        img = cap.grab()
        if img is False:
            continue
        # 推理
        bbox_arr = yolo.Inference(img, args.conf_thres, args.iou_thres)
        # 画框
        DarwBox(img, bbox_arr)


# 最小示例
def test():
    # 实列化
    yolo = Yolov5(args.half, args.image_size)
    # 设置运行设备
    yolo.SetDevice(args.device)
    # 加载pt模型
    yolo.LoadModel(args.module_path)
    # 加载图片
    img = cv2.imread("0.png")
    # 推理
    bbox_arr = yolo.Inference(img, args.conf_thres, args.iou_thres)
    DarwBox(img, bbox_arr)


if __name__ == '__main__':
    # test()
    run()
