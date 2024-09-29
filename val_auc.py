'''
    implement the feature extractions for light CNN
    @author: Alfred Xiang Wu
    @date: 2017.07.04
'''
### revised by WeiZhao Yang 2023.11.16

import logging
import torch
import os,sys
import torch.nn.parallel
import torch.optim
import torch.utils.data as data
import numpy as np
sys.path.append('.')
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve,auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from utils.BDCT import _images_to_dct, _dct_to_images
from utils.datasets.NIR_VISv2 import NIR_VISv2
from utils.datasets.LAMP_HQ import LAMP_HQ
from utils.datasets.Tufts import Tufts
from utils.model_util import distributed_concat
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from model.lightcnn_v4 import LightCNN_V4
from utils.init import get_config
import torch.nn.functional as F

label_dict = {}

def val_model(model,cfg):

    model.eval()
    global logger
    logger=logging.getLogger("eval")
    logger.setLevel(logging.INFO if int(os.environ["LOCAL_RANK"]) in [-1, 0] else logging.WARN)

    local_rank = int(os.environ["LOCAL_RANK"])
    rank1_acc = []
    vr_acc = []
    probe_features = []
    gallery_features = []
    probe_pid = []
    gallery_pid = []
    label_dict = {}
    padding = (0,0,0,0,1,1)
    datas_gallery=eval(cfg.datasets.name)(cfg.datasets.root,cfg.val.gallery,label_dict,istrain=False)
    gallery_sampler = data.distributed.DistributedSampler(datas_gallery)
    gallery_loader = data.DataLoader(datas_gallery,cfg.train.batch_size,num_workers=cfg.train.workers, pin_memory=True,sampler=gallery_sampler)
    
    for i, (image, target, _, _) in enumerate(gallery_loader):
        with torch.no_grad():
            image = image.to(local_rank)
            target = target.to(local_rank)
            server_inputs = _images_to_dct(image,cfg.train.sub_channels)
            server_img = _dct_to_images(server_inputs,cfg.train.sub_channels)
            
            output, fc = model(server_img)
            for k in range(fc.shape[0]):
                gallery_features.append(fc[k])
                gallery_pid.append(target[k])

    datas_probe=eval(cfg.datasets.name)(cfg.datasets.root,cfg.val.probe,label_dict,istrain=False)
    probe_sampler = data.distributed.DistributedSampler(datas_probe)
    probe_loader = data.DataLoader(datas_probe,cfg.train.batch_size,num_workers=cfg.train.workers, pin_memory=True,sampler=probe_sampler)
    with torch.no_grad():
        for i, (image, target, _, _) in enumerate(probe_loader):
                image = image.to(local_rank)
                target = target.to(local_rank)
                server_inputs = _images_to_dct(image,cfg.train.sub_channels)
                server_img = _dct_to_images(server_inputs,cfg.train.sub_channels)
                

                output, fc = model(server_img)
                for k in range(fc.shape[0]):
                    probe_features.append(fc[k])
                    probe_pid.append(target[k])

    probe_features = distributed_concat(torch.stack(probe_features), len(probe_features)*torch.distributed.get_world_size()).cpu().numpy()
    gallery_features = distributed_concat(torch.stack(gallery_features), len(gallery_features)*torch.distributed.get_world_size()).cpu().numpy()
    probe_pid = distributed_concat(torch.stack(probe_pid), len(probe_pid)*torch.distributed.get_world_size()).cpu().numpy()
    gallery_pid = distributed_concat(torch.stack(gallery_pid), len(gallery_pid)*torch.distributed.get_world_size()).cpu().numpy()
    
    if local_rank == 0 or -1:
        if len(gallery_pid)-len(set(gallery_pid))>0:
            duplicates = find_duplicate_indices(gallery_pid)
            for num, indices in duplicates.items():
                for j in range(1,len(indices)):
                    gallery_pid = np.delete(gallery_pid,indices[j])
                    gallery_features = np.delete(gallery_features,indices[j],axis=0)
        score = cosine_similarity(gallery_features, probe_features).T
        r_acc, tpr = compute_metric(score, probe_pid, gallery_pid)
        rank1_acc.append(r_acc)
        vr_acc.append(tpr)

        avg_r_a = np.mean(np.array(rank1_acc))
        std_r_a = np.std(np.array(rank1_acc))
        avg_v_a = np.mean(np.array(vr_acc))
        std_v_a = np.std(np.array(vr_acc))
        return avg_r_a, std_r_a, avg_v_a, std_v_a
    else:
        return 0

def compute_metric(score, probe_pid, gallery_pid):
    label = np.zeros_like(score)
    maxIndex = np.argmax(score, axis=1)
    count = 0
    for i in range(len(maxIndex)):
        probe_names_repeat = np.repeat([probe_pid[i]], len(gallery_pid), axis=0).T
        result = np.equal(probe_names_repeat, gallery_pid) * 1
        index = np.nonzero(result==1)
        if len(index[0]) != 1:
            logger.warning('more than one identity name in gallery is same as probe image name')
        else:
            label[i][index[0][0]] = 1
        
        if np.equal(int(probe_pid[i]), int(gallery_pid[maxIndex[i]])):
            count += 1
        else:
            pass
    r_acc = count/(len(probe_pid)+1e-5)
    fpr, tpr, thresholds = roc_curve(label.flatten(), score.flatten())
    
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    auc_score = auc(fpr, tpr)
    logger.info(f'rank-1 {r_acc:<7.2%} VR@FAR=1% {tpr[fpr <= 0.01][-1]:<7.2%} VR@FAR=0.1% {tpr[fpr <= 0.001][-1]:<7.2%} VR@FAR=0.01% {tpr[fpr <= 0.0001][-1]:<7.2%} AUC {auc_score:<7.2%} EER {eer:<7.2%}')
    print(f'rank-1 {r_acc:<7.2%} VR@FAR=1% {tpr[fpr <= 0.01][-1]:<7.2%} VR@FAR=0.1% {tpr[fpr <= 0.001][-1]:<7.2%} VR@FAR=0.01% {tpr[fpr <= 0.0001][-1]:<7.2%} AUC {auc_score:<7.2%} EER {eer:<7.2%}')
    return r_acc, tpr[fpr <= 0.001][-1]

def find_duplicate_indices(lst):
    seen = {}
    duplicates = {}
    for i, num in enumerate(lst):
        if num not in seen:
            seen[num] = [i]
        else:
            seen[num].append(i)
            duplicates[num] = seen[num]
    return duplicates


if __name__=="__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    cfg=get_config('config/Tufts.yml')
    
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    model=LightCNN_V4(num_classes=90)
    checkpoint = torch.load(cfg.val.weight)
    pretrained_dict = checkpoint['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    if not pretrained_dict:
        raise ValueError("Weight dict is None!")
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    try:
        val_model(model,cfg)
    except Exception as e:
        logger.error(e,exc_info=True)

# OMP_NUM_THREADS=16 torchrun --nproc_per_node 2 val_auc.py