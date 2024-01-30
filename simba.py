import torch
import torch.nn.functional as F
import utils_sima
import gol as gl
import cv2
from  torchvision import utils as vutils
import numpy as np
import test    
class SimBA:

    def __init__(self, model, dataset, image_size):
        self.model = model
        self.dataset = dataset
        self.image_size = image_size
        self.model.eval()
    
    def expand_vector(self, x, size):
        batch_size = x.size(0)
        x = x.view(-1, 3, size, size)
        z = torch.zeros(batch_size, 3, self.image_size, self.image_size)
        z[:, :, :size, :size] = x
        return z
        
    def normalize(self, x):
        return utils_sima.apply_normalization(x, self.dataset)

    def get_probs(self, x, y):
        output = self.model(self.normalize(x.cuda())).cpu()
        probs = torch.index_select(F.softmax(output, dim=-1).data, 1, y) 
        return torch.diag(probs)  
    
    def get_preds(self, x):
        output = self.model(self.normalize(x.cuda())).cpu()
        _, preds = output.data.max(1)
        return preds

    # 20-line implementation of SimBA for single image input
    def simba_single(self, x, y, num_iters=10000, epsilon=0.2, targeted=False):
        n_dims = x.view(1, -1).size(1)
        perm = torch.randperm(n_dims)
        x = x.unsqueeze(0)
        last_prob = self.get_probs(x, y)
        for i in range(num_iters):
            diff = torch.zeros(n_dims)
            diff[perm[i]] = epsilon
            left_prob = self.get_probs((x - diff.view(x.size())).clamp(0, 1), y)
            if targeted != (left_prob < last_prob):
                x = (x - diff.view(x.size())).clamp(0, 1)
                last_prob = left_prob
            else:
                right_prob = self.get_probs((x + diff.view(x.size())).clamp(0, 1), y)
                if targeted != (right_prob < last_prob):
                    x = (x + diff.view(x.size())).clamp(0, 1)
                    last_prob = right_prob
            if i % 10 == 0:
                print(last_prob)
        return x.squeeze()

    # runs simba on a batch of images <images_batch> with true labels (for untargeted attack) or target labels
    # (for targeted attack) <labels_batch>
    def simba_batch(self, images_batch, labels_batch, max_iters, freq_dims, stride, epsilon, linf_bound=0.0,
                    order='rand', targeted=False, pixel_attack=False, log_every=1):
        batch_size = images_batch.size(0)
        image_size = images_batch.size(2)
        assert self.image_size == image_size
        # sample a random ordering for coordinates independently per batch element
       ########################
        serial = gl.get_value('serial')
       ########################
        if order == 'rand':
            yolo_net = gl.get_value('net')
            serial = gl.get_value('serial')
            indices = yolo_net.detect_image(images_batch,max_iters,batch_size)[:max_iters]   
            #indices =  test. get_saliencyMaps_only(images_batch,max_iters,batch_size)[:max_iters]
            #indices = torch.randperm(3 * freq_dims * freq_dims)[:max_iters]    
        elif order == 'grad':             
            indices = utils_sima.grad(images_batch,max_iters)[:max_iters]    
        elif order == 'diag':
            indices = utils_sima.diagonal_order(image_size, 3)[:max_iters]
        elif order == 'strided':
            indices = utils_sima.block_order(image_size, 3, initial_size=freq_dims, stride=stride)[:max_iters]
        else:
            indices = utils_sima.block_order(image_size, 3)[:max_iters]
        if order == 'rand':
            expand_dims = freq_dims
        else:
            expand_dims = image_size
        n_dims = 3 * expand_dims * expand_dims  
        x = torch.zeros(batch_size, n_dims)
        # logging tensors
        probs = torch.zeros(batch_size, max_iters)  
        succs = torch.zeros(batch_size, max_iters)
        queries = torch.zeros(batch_size, max_iters)
        l2_norms = torch.zeros(batch_size, max_iters) 
        linf_norms = torch.zeros(batch_size, max_iters) 
        prev_probs = self.get_probs(images_batch, labels_batch)
        preds = self.get_preds(images_batch)
        if pixel_attack:
            trans = lambda z: z
        else:
            trans = lambda z: utils_sima.block_idct(z, block_size=image_size, linf_bound=linf_bound)
        remaining_indices = torch.arange(0, batch_size).long() 
        for k in range(max_iters):
            dim = indices[:,k]
            expanded = (images_batch[remaining_indices] + trans(self.expand_vector(x[remaining_indices], expand_dims))).clamp(0, 1)
            perturbation = trans(self.expand_vector(x, expand_dims)) 
            l2_norms[:, k] = perturbation.view(batch_size, -1).norm(2, 1)
            linf_norms[:, k] = perturbation.view(batch_size, -1).abs().max(1)[0]
            preds_next = self.get_preds(expanded) 
            preds[remaining_indices] = preds_next
            if targeted:
                remaining = preds.ne(labels_batch)
            else:
                remaining = preds.eq(labels_batch)  
            # check if all images are misclassified and stop early
            if remaining.sum() == 0:
                adv = (images_batch + trans(self.expand_vector(x, expand_dims))).clamp(0, 1)
                probs_k = self.get_probs(adv, labels_batch)
                probs[:, k:] = probs_k.unsqueeze(1).repeat(1, max_iters - k)
                succs[:, k:] = torch.ones(batch_size, max_iters - k)
                queries[:, k:] = torch.zeros(batch_size, max_iters - k)
                break
            remaining_indices = torch.arange(0, batch_size)[remaining].long()
            if k > 0:
                succs[:, k-1] = ~remaining
            diff = torch.zeros(remaining.sum(), n_dims)
            #################################
            remaining_dim = remaining_indices.numpy().astype('int32')  
            remain_num = np.shape(remaining_dim)[0]     
            for m,n in zip(range(remain_num),remaining_dim):     
                    index = dim[n].numpy().astype('int32')
                    diff[m, index] = epsilon
            #################################
            left_vec = x[remaining_indices] - diff
            right_vec = x[remaining_indices] + diff
            # trying negative direction
            adv = (images_batch[remaining_indices] + trans(self.expand_vector(left_vec, expand_dims))).clamp(0, 1)
            left_probs = self.get_probs(adv, labels_batch[remaining_indices])
            queries_k = torch.zeros(batch_size)
            # increase query count for all images
            queries_k[remaining_indices] += 1
            if targeted:
                improved = left_probs.gt(prev_probs[remaining_indices])
            else:
                improved = left_probs.lt(prev_probs[remaining_indices]) 
            # only increase query count further by 1 for images that did not improve in adversarial loss
            if improved.sum() < remaining_indices.size(0):
                queries_k[remaining_indices[~improved]] += 1
            # try positive directions
            adv = (images_batch[remaining_indices] + trans(self.expand_vector(right_vec, expand_dims))).clamp(0, 1)
            right_probs = self.get_probs(adv, labels_batch[remaining_indices])
            if targeted:
                right_improved = right_probs.gt(torch.max(prev_probs[remaining_indices], left_probs))
            else:
                right_improved = right_probs.lt(torch.min(prev_probs[remaining_indices], left_probs))
            probs_k = prev_probs.clone()
            # update x depending on which direction improved
            if improved.sum() > 0:
                left_indices = remaining_indices[improved]
                left_mask_remaining = improved.unsqueeze(1).repeat(1, n_dims)
                x[left_indices] = left_vec[left_mask_remaining].view(-1, n_dims)
                probs_k[left_indices] = left_probs[improved]
            if right_improved.sum() > 0:
                right_indices = remaining_indices[right_improved] 
                right_mask_remaining = right_improved.unsqueeze(1).repeat(1, n_dims)
                x[right_indices] = right_vec[right_mask_remaining].view(-1, n_dims)
                probs_k[right_indices] = right_probs[right_improved]
            probs[:, k] = probs_k
            queries[:, k] = queries_k
            prev_probs = probs[:, k]
            if (k + 1) % log_every == 0 or k == max_iters - 1:
                print('Iteration %d: queries = %.4f, prob = %.4f, remaining = %.4f' % (
                        k + 1, queries.sum(1).mean(), probs[:, k].mean(), remaining.float().mean()))
        expanded = (images_batch + trans(self.expand_vector(x, expand_dims))).clamp(0, 1)
        preds = self.get_preds(expanded)
        if targeted:
            remaining = preds.ne(labels_batch)
        else:
            remaining = preds.eq(labels_batch)
        succs[:, max_iters-1] = ~remaining
        ###################
        x_prub = x
        x_prub[x_prub > 0] = 255
        prub_one = torch.zeros(1, n_dims)
        for t in range(batch_size):
            #import random
            prub_one[0,:] = x_prub[t]
            prub = self.expand_vector(prub_one, expand_dims)        
            prub_file = 'save/prub_images/prub%s.jpg' % (serial * batch_size + t)       
            vutils.save_image(prub, prub_file, normalize=True)
        return expanded, probs, succs, queries, l2_norms, linf_norms