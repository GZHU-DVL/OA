from tkinter import TRUE
import torch, cv2, time, os
from PIL import Image
import os.path as osp
import numpy as np
import torch.nn.functional as F
#from argparse import ArgumentParser
from utils_Saliency  import  SalEval, Logger
from Models.SAMNet import FastSal as net
import random as rd
import gol as gl

def get_saliencyMaps(images_batch,max_iters,batch_size):
    #data_dir = './Data'
    width = 224
    height = 224
    savedir = './Outputs'
    gpu = TRUE
    pretrained = './Pretrained/SAMNet_with_ImageNet_pretrain.pth'

    if not osp.isdir(savedir):
        os.mkdir(savedir)

    model = net()
    state_dict = torch.load(pretrained,map_location=torch.device('cpu'))
    if list(state_dict.keys())[0][:7] == 'module.':
        state_dict = {key[7:]: value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    print('Model resumed from %s' % pretrained)

    if gpu:
        model = model.cuda()
    model.eval()
    mean = np.array([0.485 * 255., 0.456 * 255., 0.406 * 255.], dtype=np.float32)
    std = np.array([0.229 * 255., 0.224 * 255., 0.225 * 255.], dtype=np.float32)
    
    image_size = images_batch.size(2)
    image = torch.zeros(batch_size,3,image_size,image_size)
    for k in range(batch_size):
        image_one = images_batch[k].numpy()
        image_one = image_one *255
        image_one = image_one.astype(np.uint8)
        image_one = np.transpose(image_one,(2,1,0))
        image_one = np.transpose(image_one,(1,0,2))        
        img = cv2.cvtColor(image_one, cv2.COLOR_BGR2RGB)
        #cv2.imwrite(osp.join(savedir, '000200.jpg'),image_one)        

        image_one = Image.fromarray(img)
        #image.show()
        image_one = image_one.convert('RGB')
        ######################################
        image_one = np.array(image_one, dtype=np.float32)
        #height, width = image.shape[:2]
        image_one = (image_one - mean) / std
        image_one = cv2.resize(image_one, (width, height), interpolation=cv2.INTER_LINEAR)
        image_one = image_one.transpose((2, 0, 1))
        image_one = torch.from_numpy(image_one).unsqueeze(0)
        image[k,:]=image_one
    if gpu:
        #image, label = image.cuda(), label.cuda()
        image = image.cuda()
    # start_time = time.time()
    with torch.no_grad():
        pred = model(image)[:, 0, :, :].unsqueeze(1)

    assert pred.shape[-2:] == image.shape[-2:], '%s vs. %s' % (str(pred.shape), str(image.shape))
    pred = F.interpolate(pred, size=[height, width], mode='bilinear', align_corners=False)           
    #pred = cv2.resize(pred, (224, 224), interpolation=cv2.INTER_LINEAR)
    pred = pred.squeeze(1)
    
    #image_size = images_batch.size(2)
    pix_sali_index = torch.zeros(batch_size,image_size * image_size)
    pix_sali_index_num = torch.zeros(batch_size,1)    

    for m in range(batch_size):
        pred_one = (pred[m] * 255).cpu().numpy().astype(np.uint8)
        #pred = np.transpose(pred)
        ix_one=np.array(np.where(pred_one>20))
        #####################
        ix_image = np.zeros((224,224))
        for i in range(len(ix_one[0])):
            ix_image[ix_one[0,i],ix_one[1,i]] = 255
        serial = gl.get_value('serial') * batch_size+ m
    
        prub_name = 'saliency/%s.jpg' % (serial)
        cv2.imwrite(osp.join(savedir, prub_name), ix_image)
        pix_sali_index_one = ix_one[0] * 224 + ix_one[1]
        index_one_size = pix_sali_index_one.shape[0]

        pix_sali_index[m,0:index_one_size] = torch.from_numpy(pix_sali_index_one)
        pix_sali_index_num[m,:] = index_one_size
        ##############################
    return (pix_sali_index,pix_sali_index_num)


def random_int_list(start, stop, length):
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0
    random_list = []
    for i in range(length):
        random_list.append(rd.randint(start, stop))
    return random_list


def get_saliencyMaps_only(images_batch,max_iters,batch_size):
    #data_dir = './Data'
    width = images_batch.size(2)
    height = images_batch.size(2)
    savedir = './Outputs'
    gpu = TRUE
    pretrained = './Pretrained/SAMNet_with_ImageNet_pretrain.pth'

    if not osp.isdir(savedir):
        os.mkdir(savedir)

    model = net()
    state_dict = torch.load(pretrained,map_location=torch.device('cpu'))
    if list(state_dict.keys())[0][:7] == 'module.':
        state_dict = {key[7:]: value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    print('Model resumed from %s' % pretrained)

    if gpu:
        model = model.cuda()
    model.eval()
    mean = np.array([0.485 * 255., 0.456 * 255., 0.406 * 255.], dtype=np.float32)
    std = np.array([0.229 * 255., 0.224 * 255., 0.225 * 255.], dtype=np.float32)
    
    image_size = images_batch.size(2)
    image = torch.zeros(batch_size,3,image_size,image_size)
    for k in range(batch_size):
        image_one = images_batch[k].numpy()
        image_one = image_one *255
        image_one = image_one.astype(np.uint8)
        image_one = np.transpose(image_one,(2,1,0))
        image_one = np.transpose(image_one,(1,0,2))       
        img = cv2.cvtColor(image_one, cv2.COLOR_BGR2RGB)
        #cv2.imwrite(osp.join(savedir, '000200.jpg'),image_one)        

        image_one = Image.fromarray(img)
        #image.show()
        image_one = image_one.convert('RGB')
        ######################################
        image_one = np.array(image_one, dtype=np.float32)
        #height, width = image.shape[:2]
        image_one = (image_one - mean) / std
        image_one = cv2.resize(image_one, (width, height), interpolation=cv2.INTER_LINEAR)
        image_one = image_one.transpose((2, 0, 1))
        image_one = torch.from_numpy(image_one).unsqueeze(0)
        image[k,:]=image_one
    if gpu:
        #image, label = image.cuda(), label.cuda()
        image = image.cuda()
    # start_time = time.time()
    with torch.no_grad():
        pred = model(image)[:, 0, :, :].unsqueeze(1)

    assert pred.shape[-2:] == image.shape[-2:], '%s vs. %s' % (str(pred.shape), str(image.shape))
    pred = F.interpolate(pred, size=[height, width], mode='bilinear', align_corners=False)       
    #pred = cv2.resize(pred, (224, 224), interpolation=cv2.INTER_LINEAR)
    pred = pred.squeeze(1)
    
    #image_size = images_batch.size(2)
    pix_sali_index = torch.zeros(batch_size,max_iters)
    pix_sali_index_num = torch.zeros(batch_size,1)    

    for m in range(batch_size):
        pred_one = (pred[m] * 255).cpu().numpy().astype(np.uint8)
        #pred = np.transpose(pred)
        ix_one=np.array(np.where(pred_one>20))
        #####################
        ix_image = np.zeros((image_size,image_size))
        for i in range(len(ix_one[0])):
            ix_image[ix_one[0,i],ix_one[1,i]] = 255
        serial = gl.get_value('serial') * batch_size+ m
    
        prub_name = 'saliency/%s.jpg' % (serial)
        cv2.imwrite(osp.join(savedir, prub_name), ix_image)
        pix_sali_index_one = ix_one[0] * image_size + ix_one[1]
        index_one_size = pix_sali_index_one.shape[0]

        pix_index = [ ] 
        for n in range(index_one_size):
            num = pix_sali_index_one[n]
            pix_index.append(num)                                     
            pix_index.append(num + image_size * image_size)
            pix_index.append(num + (2 * image_size * image_size))     

        rd.shuffle(pix_index)  
        data_size = 3 * image_size * image_size - 1

        if len(pix_index) < (max_iters): 
            #insert = random_int_list(1,data_size,data_size) 
            start, stop = (int(1), int(data_size)) if 1 <= data_size else (int(1), int(data_size))
            length = int(abs(data_size)) if data_size else 0
            insert = []
            for i in range(length):
                insert.append(rd.randint(start, stop))
            ####
            for t in range(data_size):   
                rand_insert = insert[t]
                if len(pix_index) < (max_iters): 
                    if(rand_insert not in pix_index):
                        pix_index.append(rand_insert)
                else: 
                    break
        pix_index = np.array(pix_index)
        pix_index = torch.Tensor(pix_index)
        pix_index= pix_index.long()

        pix_sali_index[m,:] = pix_index[:max_iters]
        ##############################
    return (pix_sali_index)

