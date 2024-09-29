import logging
import torch
import time
import datetime
import os

import torch.nn as nn
import torch.utils.data as data
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.init import argument_parser,get_config,setup
from utils.model_util import get_model,adjust_dropout_rate,adjust_learning_rate,base_train,save_checkpoint
from utils.datasets.NIR_VISv2 import NIR_VISv2
from utils.datasets.LAMP_HQ import LAMP_HQ
from utils.datasets.Tufts import Tufts
from val_auc import val_model
from utils.file_io import PathManager


def train(model,datas_train,cfg): 
    time_train_cost = []
    time_eval_cost = []
    local_rank = int(os.environ["LOCAL_RANK"])
    train_sampler = data.distributed.DistributedSampler(datas_train)
    train_loader = data.DataLoader(datas_train,batch_size=cfg.train.batch_size,
                                    num_workers=cfg.train.workers, pin_memory=True,sampler=train_sampler)

    criterion=nn.CrossEntropyLoss()
    criterion.to(local_rank)

    optimizer=torch.optim.SGD(model.parameters(), cfg.train.lr,momentum=cfg.train.moentum,
                                weight_decay=cfg.train.weight_decay)
    stat = 0
    if cfg.train.resume:
        params_pretrain = []
        for name, value in model.named_parameters():
            if 'fc2' in name:
                params_pretrain += [{'params': value, 'lr': cfg.train.pre_lr}]
        optimizer_pretrain=torch.optim.Adam(params_pretrain, cfg.train.pre_lr, weight_decay=cfg.train.weight_decay)
        logger.info('------------start pertrain------------')
        for epoch in range(cfg.train.pre_epoch):
            train_loader.sampler.set_epoch(epoch)
            base_train(cfg, train_loader, model, criterion, optimizer_pretrain, epoch+1)

    max_rank1 = 0
    logger.info('-------------start hvi-------------')
    for epoch in range(0,cfg.train.epochs):
        start = time.perf_counter()
        train_loader.sampler.set_epoch(epoch)

        adjust_learning_rate(cfg.train.lr, optimizer, epoch, cfg.train.adjust_lr)
        adjust_dropout_rate(cfg,model,epoch)

        # train for one epoch
        base_train(cfg, train_loader, model, criterion, optimizer, epoch+1)
        end = time.perf_counter()
        time_train_cost.append(end-start)

        if epoch%cfg.train.val_frep==0:
            start = time.perf_counter()
            stat = val_model(model,cfg)
            if local_rank == 0 or -1:
                if max_rank1 <= stat[0]:
                    max_rank1 = stat[0]
                    best_model_name = cfg.datasets.name + "_" + str(stat[0]*100)[:6] + '.ckpt'
                    save_path = os.path.join(cfg.logs.output_path, best_model_name)
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'model': cfg.train.model,
                        'state_dict': model.module.state_dict(),
                        'stat': stat,
                        'time' : time.time()
                    }, save_path)
            end = time.perf_counter()
            time_eval_cost.append(end-start)
            etc = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time() + sum(time_train_cost)/len(time_train_cost)*(cfg.train.epochs-epoch) 
                                                                    + sum(time_eval_cost)/len(time_eval_cost)*int((cfg.train.epochs-epoch)/cfg.train.val_frep)))
            logger.info(f"etc: {etc}")
    if local_rank == 0 or -1:
        logger.info(f"The best Rank-1 is {max_rank1}")
        best_model_path = os.path.join('best_weight', str(datetime.date.today()))
        if not PathManager.exists(best_model_path):
            PathManager.mkdirs(best_model_path)
        PathManager.copy(save_path,os.path.join(best_model_path,best_model_name),overwrite=True)

    return max_rank1


if __name__=="__main__":
    parser=argument_parser()
    args=parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])

    cfg_path = args.config_file
    cfg=get_config(cfg_path)
    cfg.defrost()
    cfg.logs.output_path = os.path.join(cfg.logs.output_path, str(datetime.date.today()),"")
    cfg.freeze()
    setup(cfg,args)
    logger=logging.getLogger("hvi")
    
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)

    model,datas_train=get_model(cfg,protocols=cfg.train.protocols,resume=cfg.train.resume)
    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    try:
        max_rank1 = train(model,datas_train,cfg)
    except Exception as e:
        logger.error(e,exc_info=True)
        
# OMP_NUM_THREADS=16 torchrun --nproc_per_node 2 hvi.py