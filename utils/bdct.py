from torchjpeg import dct
from torch.nn import functional as F
import torch, os

def _images_to_dct(x, sub_channels=None, size=8, stride=8, pad=0, dilation=1):

    x = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=True)
    x *= 255
    x = dct.to_ycbcr(x)
    x -= 128
    bs, ch, h, w = x.shape
    block_num = h // stride
    x = x.view(bs * ch, 1, h, w)
    x = F.unfold(x, kernel_size=(size, size), dilation=dilation, padding=pad,stride=(stride, stride))
    x = x.transpose(1, 2)
    x = x.view(bs, ch, -1, size, size)
    dct_block = dct.block_dct(x)
    dct_block = dct_block.view(bs, ch, block_num, block_num, size * size).permute(0, 1, 4, 2, 3)

    channels = list(set([i for i in range(64)]) - set(sub_channels))
    main_inputs = dct_block[:, :, channels, :, :]
    main_inputs = main_inputs.reshape(bs, -1, block_num, block_num)
    return main_inputs



def _dct_to_images(main_inputs, sub_channels=None, size=8, stride=8, pad=0, dilation=1):

    bs, _, _, _ = main_inputs.shape
    sampling_rate = 8

    full_inputs = torch.zeros((bs, 3, 64, 16 * sampling_rate, 16 * sampling_rate))
    full_inputs = full_inputs.to(main_inputs.device)
    remaining_channels = list(set(range(64)) - set(sub_channels))
    main_inputs = main_inputs.view(bs, 3, len(remaining_channels), 16 * sampling_rate, 16 * sampling_rate)
    full_inputs[:, :, remaining_channels, :, :] = main_inputs
    full_inputs = full_inputs.permute(0, 1, 3, 4, 2)
    full_inputs = full_inputs.view(bs, 3, 16 * 16 * sampling_rate * sampling_rate, 8, 8)
    x = dct.block_idct(full_inputs)
    
    x = x.view(bs * 3, 16 * 16 * sampling_rate * sampling_rate, 64)
    x = x.transpose(1, 2)
    x = F.fold(x, output_size=(128 * sampling_rate, 128 * sampling_rate),
               kernel_size=(size, size), dilation=dilation, padding=pad, stride=(stride, stride))
    x = x.view(bs, 3, 128 * sampling_rate, 128 * sampling_rate)
    x += 128
    x = dct.to_rgb(x)
    x /= 255
    x = F.interpolate(x, scale_factor=1 / sampling_rate, mode='bilinear', align_corners=True)
    return x