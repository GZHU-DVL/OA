import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.models as models
import utils_sima as utils
import math
import random
import argparse
import os
from simba import SimBA
import json
from  torchvision import utils as vutils
from yolo import YOLO
import gol as gl
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Runs SimBA on a set of images')
parser.add_argument('--data_root', type=str, default='/home/syzx/ZC/object-attation-attack-master',required=True, help='root directory of imagenet data')
parser.add_argument('--result_dir', type=str, default='save', help='directory for saving results')
parser.add_argument('--sampled_image_dir', type=str, default='save', help='directory to cache sampled images')
parser.add_argument('--model', type=str, default='resnet50', help='type of base model to use')
parser.add_argument('--num_runs', type=int, default=1000, help='number of image samples')
parser.add_argument('--batch_size', type=int, default=1, help='batch size for parallel runs')
parser.add_argument('--num_iters', type=int, default=10000, help='maximum number of iterations, 0 for unlimited')
parser.add_argument('--log_every', type=int, default=10, help='log every n iterations')
parser.add_argument('--epsilon', type=float, default=0.2, help='step size per iteration')
parser.add_argument('--linf_bound', type=float, default=0.0, help='L_inf bound for frequency space attack')
parser.add_argument('--freq_dims', type=int, default=224, help='dimensionality of 2D frequency space')
parser.add_argument('--order', type=str, default='rand', help='(random) order of coordinate selection')
#parser.add_argument('--order', type=str, default='strided', help='(random) order of coordinate selection')
parser.add_argument('--stride', type=int, default=7, help='stride for block order')
parser.add_argument('--targeted', action='store_true', help='perform targeted attack')
parser.add_argument('--pixel_attack',default='pixel_attack', action='store_true', help='attack in pixel space')
#parser.add_argument('--pixel_attack', action='store_true', help='attack in pixel space')
parser.add_argument('--save_suffix', type=str, default='', help='suffix appended to save file')
args = parser.parse_args()


if not os.path.exists(args.result_dir):
    os.mkdir(args.result_dir)
if not os.path.exists(args.sampled_image_dir):
    os.mkdir(args.sampled_image_dir)

# load model and dataset
model = getattr(models, args.model)(pretrained=True).cuda()
model.eval()
if args.model.startswith('inception'):
    image_size = 299
    testset = dset.ImageFolder(args.data_root + '/val', utils.INCEPTION_TRANSFORM)
else:
    image_size = 224
    #testset = dset.ImageFolder(args.data_root + '/val', utils.IMAGENET_TRANSFORM)
attacker = SimBA(model, 'imagenet', image_size)

# load sampled images or sample new ones
# this is to ensure all attacks are run on the same set of correctly classified images
batchfile = '%s/images_%s_%d.pth' % (args.sampled_image_dir, args.model, args.num_runs)
if os.path.isfile(batchfile):
    checkpoint = torch.load(batchfile)
    images = checkpoint['images']
    labels = checkpoint['labels']
else:
    images = torch.zeros(args.num_runs, 3, image_size, image_size)
    labels = torch.zeros(args.num_runs).long()
    preds = labels + 1
    while preds.ne(labels).sum() > 0:
        idx = torch.arange(0, images.size(0)).long()[preds.ne(labels)]
        for i in list(idx):
            images[i], labels[i] = testset[random.randint(0, len(testset) - 1)]
        preds[idx], _ = utils.get_preds(model, images[idx], 'imagenet', batch_size=args.batch_size)
    torch.save({'images': images, 'labels': labels}, batchfile)

if args.order == 'rand':
    n_dims = 3 * args.freq_dims * args.freq_dims
else:
    n_dims = 3 * image_size * image_size            
if args.num_iters > 0:
    max_iters = int(min(n_dims, args.num_iters))    
else:
    max_iters = int(n_dims)
N = int(math.floor(float(args.num_runs) / float(args.batch_size)))    

#############
yolo = YOLO()
gl._init()
gl.set_value('net', yolo)

#############
for i in range(N):
    upper = min((i + 1) * args.batch_size, args.num_runs)
    images_batch = images[(i * args.batch_size):upper]     
    labels_batch = labels[(i * args.batch_size):upper]     
    # replace true label with random target labels in case of targeted attack
    if args.targeted:
        labels_targeted = labels_batch.clone()
        while labels_targeted.eq(labels_batch).sum() > 0:
            labels_targeted = torch.floor(1000 * torch.rand(labels_batch.size())).long()
        labels_batch = labels_targeted
###################
    gl.set_value('serial', i)
###################
    adv, probs, succs, queries, l2_norms, linf_norms = attacker.simba_batch(
        images_batch, labels_batch, max_iters, args.freq_dims, args.stride, args.epsilon, linf_bound=args.linf_bound,
        order=args.order, targeted=args.targeted, pixel_attack=args.pixel_attack, log_every=args.log_every)
    if i == 0:
        all_adv = adv
        all_probs = probs
        all_succs = succs
        all_queries = queries
        all_l2_norms = l2_norms
        all_linf_norms = linf_norms
    else:
        all_adv = torch.cat([all_adv, adv], dim=0)
        all_probs = torch.cat([all_probs, probs], dim=0)
        all_succs = torch.cat([all_succs, succs], dim=0)
        all_queries = torch.cat([all_queries, queries], dim=0)

        all_l2_norms = torch.cat([all_l2_norms, l2_norms], dim=0)
        all_linf_norms = torch.cat([all_linf_norms, linf_norms], dim=0)
    ###
    #queries_single = queries.sum()
    ####

    if args.pixel_attack:
        prefix = 'pixel'
    else:
        prefix = 'dct'
    if args.targeted:
        prefix += '_targeted'
    
    
    avg_l2_norm_row = torch.max(all_l2_norms,1)[0].data 
    avg_l2_norm = avg_l2_norm_row.sum().item()
    avg_l2_norm = avg_l2_norm / avg_l2_norm_row.size(0)

    avg_linf_norm_row = torch.max(all_linf_norms,1)[0].data
    avg_linf_norm = avg_linf_norm_row.sum().item()
    avg_linf_norm = avg_linf_norm / avg_linf_norm_row.size(0)

    #avg_queries_row = torch.max(all_queries,1)[0].data
    avg_queries = all_queries.sum().item()
    avg_queries = avg_queries / all_queries.size(0)

    avg_succs_row = torch.max(all_succs,1)[0].data
    avg_succs = avg_succs_row.sum().item()
    avg_succs = avg_succs / avg_succs_row.size(0)


    #######
        #original_img_file = 'save/original_images_Dct/original%s.jpg' % (i)   
    for t in range(args.batch_size):    
        original_img_file = 'save/original_images/original%s.jpg' % (i * args.batch_size + t)       
        vutils.save_image(images_batch[t], original_img_file, normalize=True)
        #adv_img_file = 'save/adv_images_Dct/adv%s.jpg' % (i)     
        adv_img_file = 'save/adv_images/adv%squeries%s.jpg' % (i * args.batch_size + t, queries[t].sum())       
        vutils.save_image(adv[t], adv_img_file, normalize=True)  

    print('avg_queries = %.4f, avg_l2_norm = %.4f, avg_linf_norm = %.4f, avg_succs = %.4f' % (
        avg_queries, avg_l2_norm, avg_linf_norm, avg_succs))
    #file_handle=open('log_Dct.txt',mode='w') 
    file_handle=open('log.txt',mode='w')    
    file_handle.write('avg_queries = %.4f, avg_l2_norm = %.4f, avg_linf_norm = %.4f, avg_succs = %.4f \n' % (
        avg_queries, avg_l2_norm, avg_linf_norm, avg_succs))
    file_handle.close()
    ##########

    savefile = '%s/%s_%s_%d_%d_%d_%.4f_%s%s.pth' % (
        args.result_dir, prefix, args.model, args.num_runs, args.num_iters, args.freq_dims, args.epsilon, args.order, args.save_suffix)
    torch.save({'adv': all_adv, 'probs': all_probs, 'succs': all_succs, 'queries': all_queries,
                'l2_norms': all_l2_norms, 'linf_norms': all_linf_norms}, savefile)
file_handle.close()
