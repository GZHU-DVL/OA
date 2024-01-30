#-------------------------------------#
#       CREATE YOLO
#-------------------------------------#
import colorsys
import os
import time
import cv2
import numpy as np
from numpy.lib.shape_base import row_stack
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
#import  mmcv
#from nets.yolo4 import YoloBody
from yolo4  import YoloBody
from utils import (DecodeBox, letterbox_image, non_max_suppression,
                         yolo_correct_boxes)
import random as rd
import test

class YOLO(object):
    _defaults = {
        "model_path"        : 'model_data/yolo4_weights.pth',
        "anchors_path"      : 'model_data/yolo_anchors.txt',
        "classes_path"      : 'model_data/coco_classes.txt',
        "model_image_size"  : (416, 416, 3),
        "confidence"        : 0.05,
        "iou"               : 0.3,
        "cuda"              :False ,

        "letterbox_image"   : False,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"


    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.generate()


    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names
    

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape([-1, 3, 2])[::-1,:,:]


    def generate(self):

        self.net = YoloBody(len(self.anchors[0]), len(self.class_names)).eval()

        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(self.model_path, map_location=device)
        self.net.load_state_dict(state_dict)
        print('Finished!')
        
        if not self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()


        self.yolo_decodes = []
        for i in range(3):
            self.yolo_decodes.append(DecodeBox(self.anchors[i], len(self.class_names),  (self.model_image_size[1], self.model_image_size[0])))

        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

#######################################
    def random_int_list(self,start, stop, length):
        start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
        length = int(abs(length)) if length else 0
        random_list = []
        for i in range(length):
            random_list.append(rd.randint(start, stop))
        return random_list

    def detect_image(self,image_batch,max_iters,batch_size):
        pix_sali_index,pix_sali_indexnum =  test. get_saliencyMaps(image_batch,max_iters,batch_size)
        #############
        pix_yolo_index = torch.zeros(batch_size,max_iters)      
        #images = torch.zeros(batch_size,3,self.model_image_size[1],self.model_image_size[0])
        for k in range(batch_size):
            image = image_batch[k]
            image = image.numpy()
            image = image *255
            image = image.astype(np.uint8)
            image = np.transpose(image,(2,1,0))

            image = np.transpose(image,(1,0,2)) 

            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(img)
            image = image.convert('RGB')
            #image = Image.fromarray(image.astype('unit8')).convert('RGB')

            image_shape = np.array(np.shape(image)[0:2])
            image_size = image_shape[0]
            pix_one_channel = image_size * image_size       

            if self.letterbox_image:
                crop_img = np.array(letterbox_image(image, (self.model_image_size[1],self.model_image_size[0])))
            else:
                crop_img = image.resize((self.model_image_size[1],self.model_image_size[0]), Image.BICUBIC)
            photo = np.array(crop_img,dtype = np.float32) / 255.0
            photo = np.transpose(photo, (2, 0, 1))
            #photo = torch.from_numpy(np.asarray(photo))
            images =  [photo]

            with torch.no_grad():
                images = torch.from_numpy(np.asarray(images))
                images = images.cuda()
                outputs = self.net(images)
                output_list = []
                for i in range(3):
                    output_list.append(self.yolo_decodes[i](outputs[i]))
                output = torch.cat(output_list, 1)
                batch_detections = non_max_suppression(output, len(self.class_names),
                                                        conf_thres=self.confidence,
                                                        nms_thres=self.iou)
                try:
                    batch_detections = batch_detections[0].cpu().numpy()
                except:
                    pix_yolo_index[k,:] =  torch.randperm(3 * image_shape[0] * image_shape[0])[:max_iters]   
                    continue    

                top_index = batch_detections[:,4] * batch_detections[:,5] > self.confidence
                #top_conf = batch_detections[top_index,4]*batch_detections[top_index,5]
                top_label = np.array(batch_detections[top_index,-1],np.int32)
                top_bboxes = np.array(batch_detections[top_index,:4])
                top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:,0],-1),np.expand_dims(top_bboxes[:,1],-1),np.expand_dims(top_bboxes[:,2],-1),np.expand_dims(top_bboxes[:,3],-1)

                if self.letterbox_image:
                    boxes = yolo_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([self.model_image_size[0],self.model_image_size[1]]),image_shape)
                else:
                    top_xmin = top_xmin / self.model_image_size[1] * image_shape[1]
                    top_ymin = top_ymin / self.model_image_size[0] * image_shape[0]
                    top_xmax = top_xmax / self.model_image_size[1] * image_shape[1]
                    top_ymax = top_ymax / self.model_image_size[0] * image_shape[0]
                    boxes = np.concatenate([top_ymin,top_xmin,top_ymax,top_xmax], axis=-1)
                
            pix_index = [ ]       
            pix_index_yololeft = [ ] 
            top_aix = image_size
            left_aix = image_size
            bottom_aix = 0
            right_aix = 0

            for i, c in enumerate(top_label): 
                top, left, bottom, right = boxes[i]
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
                right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))
                if(top < top_aix):
                    top_aix = top
                if(bottom > bottom_aix):
                    bottom_aix = bottom           
                if(left < left_aix):
                    left_aix = left
                if(right > right_aix):
                    right_aix = right
            ares =( bottom_aix - top_aix)* (right_aix - left_aix)
            sali_num = pix_sali_indexnum[k].numpy()[0].astype('int32')
            #####################################
            '''yolo_rate = pix_one_channel  /  ares 
            if (yolo_rate > 20):
                return torch.randperm(3 * image_shape[0] * image_shape[0]) ''' 
        #####################################
            for  row in range(top_aix,bottom_aix,1):
                for  conl in range(left_aix,right_aix,1):           
                    #num = (row * image_size) + conl     
                    num = (row * image_size) + conl 
                    if (num in pix_sali_index[k]) and ((ares / sali_num) <4):
                        pix_index.append(num)                                      
                        pix_index.append(num + pix_one_channel)
                        pix_index.append(num + (2 * pix_one_channel))
                    else: 
                        pix_index_yololeft.append(num)                    
                        pix_index_yololeft.append(num + pix_one_channel)
                        pix_index_yololeft.append(num + (2 * pix_one_channel))       
        
            rd.shuffle(pix_index)  
            rd.shuffle(pix_index_yololeft)  

            #pix_index.append(pix_index_yololeft)
            pix_index_result = np.hstack([pix_index,pix_index_yololeft])        
            pix_index_result = np.array(pix_index_result)
            #grand_index_num = len(grand_index)
            data_size = 3 * pix_one_channel -1
            
            if len(pix_index_result) < (max_iters): 
                #insert = random_int_list(1,data_size,data_size)
                ####
                start, stop = (int(1), int(data_size)) if 1 <= data_size else (int(1), int(data_size))
                length = int(abs(data_size)) if data_size else 0
                insert = []
                for i in range(length):
                    insert.append(rd.randint(start, stop))
                ####
                for t in range(data_size):   
                    rand_insert = insert[t]
                    if len(pix_index_result) < (max_iters): 
                        if(rand_insert not in pix_index_result):
                            pix_index_result = np.append(pix_index_result,rand_insert)
                    else: 
                        break

            pix_index_result = np.array(pix_index_result)
            pix_index_result = torch.Tensor(pix_index_result)
            pix_index_result= pix_index_result.long()
            pix_yolo_index[k,:] = pix_index_result[:max_iters]
        return pix_yolo_index


    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])

        if self.letterbox_image:
            crop_img = np.array(letterbox_image(image, (self.model_image_size[1],self.model_image_size[0])))
        else:
            crop_img = image.convert('RGB')
            crop_img = crop_img.resize((self.model_image_size[1],self.model_image_size[0]), Image.BICUBIC)
        photo = np.array(crop_img,dtype = np.float32) / 255.0
        photo = np.transpose(photo, (2, 0, 1))
        images = [photo]

        with torch.no_grad():
            images = torch.from_numpy(np.asarray(images))
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            output_list = []
            for i in range(3):
                output_list.append(self.yolo_decodes[i](outputs[i]))
            output = torch.cat(output_list, 1)
            batch_detections = non_max_suppression(output, len(self.class_names),
                                                    conf_thres=self.confidence,
                                                    nms_thres=self.iou)
            try:
                batch_detections = batch_detections[0].cpu().numpy()
                top_index = batch_detections[:,4]*batch_detections[:,5] > self.confidence
                top_conf = batch_detections[top_index,4]*batch_detections[top_index,5]
                top_label = np.array(batch_detections[top_index,-1],np.int32)
                top_bboxes = np.array(batch_detections[top_index,:4])
                top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:,0],-1),np.expand_dims(top_bboxes[:,1],-1),np.expand_dims(top_bboxes[:,2],-1),np.expand_dims(top_bboxes[:,3],-1)
                
                if self.letterbox_image:
                    boxes = yolo_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([self.model_image_size[0],self.model_image_size[1]]),image_shape)
                else:
                    top_xmin = top_xmin / self.model_image_size[1] * image_shape[1]
                    top_ymin = top_ymin / self.model_image_size[0] * image_shape[0]
                    top_xmax = top_xmax / self.model_image_size[1] * image_shape[1]
                    top_ymax = top_ymax / self.model_image_size[0] * image_shape[0]
                    boxes = np.concatenate([top_ymin,top_xmin,top_ymax,top_xmax], axis=-1)
            
            except:
                pass
                
        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                outputs = self.net(images)
                output_list = []
                for i in range(3):
                    output_list.append(self.yolo_decodes[i](outputs[i]))
                output = torch.cat(output_list, 1)
                batch_detections = non_max_suppression(output, len(self.class_names),
                                                        conf_thres=self.confidence,
                                                        nms_thres=self.iou)
                try:
                    batch_detections = batch_detections[0].cpu().numpy()
                    top_index = batch_detections[:,4]*batch_detections[:,5] > self.confidence
                    top_conf = batch_detections[top_index,4]*batch_detections[top_index,5]
                    top_label = np.array(batch_detections[top_index,-1],np.int32)
                    top_bboxes = np.array(batch_detections[top_index,:4])
                    top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:,0],-1),np.expand_dims(top_bboxes[:,1],-1),np.expand_dims(top_bboxes[:,2],-1),np.expand_dims(top_bboxes[:,3],-1)
                   
                    if self.letterbox_image:
                        boxes = yolo_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([self.model_image_size[0],self.model_image_size[1]]),image_shape)
                    else:
                        top_xmin = top_xmin / self.model_image_size[1] * image_shape[1]
                        top_ymin = top_ymin / self.model_image_size[0] * image_shape[0]
                        top_xmax = top_xmax / self.model_image_size[1] * image_shape[1]
                        top_ymax = top_ymax / self.model_image_size[0] * image_shape[0]
                        boxes = np.concatenate([top_ymin,top_xmin,top_ymax,top_xmax], axis=-1)
                
                except:
                    pass

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

