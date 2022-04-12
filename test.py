# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from data import test_dataset
from imageio import imsave

from model import MyNet


def inference():
    model = MyNet()
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda:0" if USE_CUDA else "cpu")
    model = nn.DataParallel(model, device_ids=[0, 1])
    model.to(device)

    model.load_state_dict(torch.load('./best.pth'), strict=False)
    model.cuda()
    model.eval()

    test_loader = test_dataset('./1-141.png', 352)
    image = test_loader.load_data()
    image = image.cuda()

    lateral_map_1 = model(image)
    res = lateral_map_1
    # 还原为原来的尺寸
    res = F.upsample(res, size=(352,352), mode='bilinear', align_corners=False)
    res = res.sigmoid().data.cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)

    # np.set_printoptions(threshold=np.inf)
    imsave('./1.png', (res*255).astype(np.uint8))


if __name__ == "__main__":
    inference()