import os
import logging
import torch
from torchvision import utils as v_utils
from utils.dra import DRA
from utils.init import argument_parser,get_config,setup
import datetime
import numpy as np
import torch.distributed as dist
import torch.utils.data as data
from utils.model_util import get_model
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.BDCT import _images_to_dct,_dct_to_images

logger=logging.getLogger("dra")
logger.setLevel(logging.INFO if int(os.environ["LOCAL_RANK"]) in [-1, 0] else logging.WARN)

def main(model,datas_train,cfg):
    local_rank = int(os.environ["LOCAL_RANK"])
    data_sampler = data.distributed.DistributedSampler(datas_train)
    data_loader = data.DataLoader(datas_train,batch_size=cfg.avih.batch_size,
                                    num_workers=cfg.train.workers, pin_memory=True,sampler=data_sampler)

    avh = DRA(cfg)



    for i, (img, _, _, name) in enumerate(data_loader):
        logger.info(f'---{i}---')
        img = img.to(local_rank)
        server_inputs= _images_to_dct(img,cfg.train.sub_channels)
        server_img = _dct_to_images(server_inputs,cfg.train.sub_channels)
        # Encrypted images
        att_list = avh.attack(server_img, model,adv_size=cfg.avih.adv_size)
        for i in range(len(att_list)):
            att = att_list[i]
            att_save1 = (att*255).cpu().detach().numpy().astype(np.uint8)
            for j in range(len(name)):
                os.makedirs(os.path.join(cfg.logs.output_path,f'{200*(i+1)}','/'.join(name[j].split(' ')[0].split('/')[:-1])),exist_ok=True)
                np.save(cfg.logs.output_path + f'{200*(i+1)}/' + name[j].split(' ')[0].split('.')[0],att_save1[j])
                v_utils.save_image(att[j],cfg.logs.output_path + f'{200*(i+1)}/'  + name[j].split(' ')[0].split('.')[0]+'.jpg',normalize=True)

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
    
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)

    model,datas_train=get_model(cfg,resume=cfg.avih.resume,protocols=cfg.avih.protocols)
    model = model.to(local_rank).eval()
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)
    try:
        max_rank1 = main(model,datas_train,cfg)
    except Exception as e:
        logger.error(e,exc_info=True)