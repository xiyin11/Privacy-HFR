import os
import math
import torch
import logging
from utils.datasets.LAMP_HQ import LAMP_HQ
from utils.datasets.NIR_VISv2 import NIR_VISv2
from utils.datasets.Tufts import Tufts
import torch.backends.cudnn as cudnn
from model.lightcnn_v4 import LightCNN_V4
from utils.BDCT import _images_to_dct , _dct_to_images
import torch.nn.functional as F



logger=logging.getLogger('hvi')


def save_checkpoint(state,filename):
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0
        self.avg   = 0
        self.sum   = 0
        self.count = 0

    def update(self, val, n=1):
        self.val   = val
        self.sum   += val * n
        self.count += n
        self.avg   = self.sum / self.count


def adjust_learning_rate(lr,optimizer, epoch ,step  = 20):
    scale = 0.457305051927326
    lr = lr * (scale ** (epoch // step))
    if (epoch != 0) and (epoch % step == 0):
        logger.info(f'Change lr to: {lr}')
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * scale
    return lr

def adjust_dropout_rate(cfg,model,epoch):
    dropout_rates = cfg.train.dropout_rate
    step = cfg.train.adjust_dropout
    rate = dropout_rates ** (0.7 ** (epoch // step)) if dropout_rates ** (0.7 ** (epoch // step))<=cfg.train.max_dropout else cfg.train.max_dropout 
    if (epoch != 0) and (epoch % step == 0):
        logger.info(f'Change dropout to {rate}')
        model.module.dropout_rate = rate
    return rate
     
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred    = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def get_model(cfg,resume=None,protocols=None):
    cudnn.benchmark=True
    datas_train=eval(cfg.datasets.name)(cfg.datasets.root,protocols)
    model=LightCNN_V4(num_classes=300,dropout_rate=cfg.train.dropout_rate)

    # load pretrained lightcnn
    if resume:
        logger.info("loading pretrained model '{}'".format(resume))
        checkpoint = torch.load(resume)
        pretrained_dict = checkpoint['state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        if not pretrained_dict:
            raise ValueError("Weight dict is None!")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model,datas_train

def base_train(cfg,train_loader, model, criterion, optimizer, epoch):
    losses     = AverageMeter()
    top1       = AverageMeter()
    top5       = AverageMeter()
    model.train()
    local_rank = int(os.environ['LOCAL_RANK'])
    for i, (image, target, _, _) in enumerate(train_loader):
        target = torch.Tensor(target)

        image = image.to(local_rank)
        target = target.to(local_rank)

        server_inputs = _images_to_dct(image,cfg.train.sub_channels)
        server_img = _dct_to_images(server_inputs,cfg.train.sub_channels)
        output, fc = model(server_img)

        # compute output
        loss   = criterion(output, target)


        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))
        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))
        top5.update(prec5.item(), image.size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if math.isnan(loss):
            raise ValueError("The loss value is NaN!")
        if i % cfg.logs.print_freq == 1:
            logger.info(
                'Epoch: [{0}][{1:3}/{2:3}]\t'
                'Loss {loss.avg:>8.4f}\t'
                'Prec@1 {top1.avg:>6.3f}\t'
                'Prec@5 {top5.avg:>6.3f}\t'
                .format(
                   epoch, i, len(train_loader),
                    loss=losses, top1=top1, top5=top5))

def distributed_concat(tensor, num_total_examples):
    output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    return concat[:num_total_examples]