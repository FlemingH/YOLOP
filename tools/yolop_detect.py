# modify from demo.py

import os, sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
import torchvision.transforms as transforms

from lib.utils.utils import select_device, time_synchronized
from lib.models import get_net
from lib.core.general import non_max_suppression, scale_coords
from lib.utils import plot_one_box,show_seg_result
from lib.core.function import AverageMeter
from lib.core.postprocess import morphological_process
normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

def letterbox_for_img(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))


    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding

    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def detect(cfg, img0):

    device = select_device(None, 'cpu')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = get_net(cfg)
    checkpoint = torch.load('../weights/End-to-end.pth', map_location= device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    if half:
        model.half()  # to FP16

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    img = torch.zeros((1, 3, 640, 640), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    model.eval()

    inf_time = AverageMeter()
    nms_time = AverageMeter()

    # set True to speed up constant image size inference
    cudnn.benchmark = True  
    
    # LoadImages
    h0, w0 = img0.shape[:2]
    img, ratio, pad = letterbox_for_img(img0, new_shape=640, auto=True)
    h, w = img.shape[:2]
    shapes = (h0, w0), ((h / h0, w / w0), pad)
    img = np.ascontiguousarray(img)


    img = transform(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    # Inference
    t1 = time_synchronized()
    det_out, da_seg_out,ll_seg_out= model(img)
    t2 = time_synchronized()
    inf_out, _ = det_out
    inf_time.update(t2-t1,img.size(0))

    # Apply NMS
    t3 = time_synchronized()
    det_pred = non_max_suppression(inf_out, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False)
    t4 = time_synchronized()

    nms_time.update(t4-t3,img.size(0))
    det=det_pred[0]

    _, _, height, width = img.shape
    h,w,_=img0.shape
    pad_w, pad_h = shapes[1][1]
    pad_w = int(pad_w)
    pad_h = int(pad_h)
    ratio = shapes[1][0][1]

    da_predict = da_seg_out[:, :, pad_h:(height-pad_h),pad_w:(width-pad_w)]
    da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=int(1/ratio), mode='bilinear')
    _, da_seg_mask = torch.max(da_seg_mask, 1)
    da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()

    
    ll_predict = ll_seg_out[:, :,pad_h:(height-pad_h),pad_w:(width-pad_w)]
    ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=int(1/ratio), mode='bilinear')
    _, ll_seg_mask = torch.max(ll_seg_mask, 1)
    ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()

    img0 = show_seg_result(img0, (da_seg_mask, ll_seg_mask), _, _, is_demo=True)

    if len(det):
        det[:,:4] = scale_coords(img.shape[2:],det[:,:4],img0.shape).round()
        for *xyxy,conf,cls in reversed(det):
            label_det_pred = f'{names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, img0 , label=label_det_pred, color=colors[int(cls)], line_thickness=2)

    return img0
