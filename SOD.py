# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 20:41:08 2021

@author: xfc
"""

import torch, cv2, time, os
from PIL import Image
import os.path as osp
import numpy as np
import torch.nn.functional as F
from Models.SAMNet import FastSal as net
       

class SOD:
    
    def __init__(self, img,crop_h,crop_w):
        
        crop_h=224
        crop_w=224
        self.mean= np.array([0.485 * 255., 0.456 * 255., 0.406 * 255.], dtype=np.float32)
        self.std = np.array([0.229 * 255., 0.224 * 255., 0.225 * 255.], dtype=np.float32)
        self.model = net()
        self.net_path='./Pretrained/SAMNet_with_ImageNet_pretrain.pth'
        state_dict = torch.load(self.net_path,map_location=torch.device('cpu'))
        if list(state_dict.keys())[0][:7] == 'module.':
            state_dict = {key[7:]: value for key, value in state_dict.items()}
        self.model.load_state_dict(state_dict, strict=True)
        # if gpu:
        #     model = model.cuda()
        self.model.eval()

    def __call__(self, img,crop_h,crop_w):
        img=img.astype('float32')
        height, width = img.shape[:2]
        img = (img - self.mean) / self.std
        img = cv2.resize(img, (336, 336), interpolation=cv2.INTER_LINEAR)
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0)
        with torch.no_grad():
            pred = self.model(img)[:, 0, :, :].unsqueeze(1)
        assert pred.shape[-2:] == img.shape[-2:], '%s vs. %s' % (str(pred.shape), str(img.shape))
        pred = F.interpolate(pred, size=[height, width], mode='bilinear', align_corners=False)
        pred = pred.squeeze(1)
        pred = (pred[0] * 255).cpu().numpy().astype(np.uint8)
        ix=np.array(np.where(pred>100))
        box=(ix[1].min(),ix[0].min(),ix[1].max(),ix[0].max())
        region=img.crop(box)
        
    
    
    

