import os
import torch
import logging
import torch.nn.functional as F

logger=logging.getLogger('dra')

class DRA():
    def __init__(self, cfg):
        self.cfg = cfg
        self.alpha = 0.01

    def initialization(self, data):
        adv = torch.randn_like(data)
        return adv

    def get_feature(self, model, data):
        _, fea = model.forward(data)
        return fea

    def attack(self, data, model,adv_size=[128,128]):
        local_rank = int(os.environ["LOCAL_RANK"])
        cfg = self.cfg
        # initialization
        adv = self.initialization(torch.zeros(data.size(0), 1, adv_size[0],adv_size[1]))
        adv = adv.detach().clone().requires_grad_(True)

        sum_grad = torch.zeros_like(adv).to(local_rank)
        alpha = cfg.avih.alpha
        tmp_losses = []
        num_lo = 0
        feature = self.get_feature(model, data)
        att_list = []

        for i in range(cfg.avih.num_iter):
            adv = adv.detach().clone().to(local_rank).requires_grad_(True)
            padding = (64-adv_size[1]//2, 64-adv_size[1]//2, 64-adv_size[0]//2, 64-adv_size[0]//2,1,1)
            adv_inputs = F.pad(adv, padding, mode='constant', value=0)

            adv_feature = self.get_feature(model, adv_inputs.to(local_rank))
            model.zero_grad()

            loss_f_i = torch.mean((adv_feature - feature) ** 2)
            loss = loss_f_i
            loss.backward(retain_graph=True)
            grad = adv.grad.data.clone()

            grad = grad / grad.abs().mean(dim=[1, 2, 3], keepdim=True)
            sum_grad = 0.3 * sum_grad + grad
            adv.grad.data.zero_()
            adv = adv.data.clone()
            adv = adv - sum_grad * alpha
            adv = torch.clamp(adv, min=0, max=1)

            tmp_losses.append(loss.data.unsqueeze(0).cpu().detach().numpy())
            if i > 1 and tmp_losses[i] > tmp_losses[i - 1]:
                num_lo = num_lo + 1
                if num_lo == 16:
                    alpha = alpha * 0.85
                    num_lo = 0
            if i % 60 == 0 or i == cfg.avih.num_iter - 1 or i == 1000 - 1:
                logger.info(f'loss: {loss.cpu().detach().numpy()}')

            if i%200 == 0 and i!=0:
                att_list.append(adv)
        return att_list

