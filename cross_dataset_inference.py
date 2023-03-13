import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

from multiview_detector.datasets import *
from multiview_detector.evaluation.evaluate import evaluate
from multiview_detector.models.ca_sa_detector import CA_SA_Detector
from multiview_detector.models.persp_trans_detector import PerspTransDetector
from multiview_detector.utils.image_utils import img_color_denormalize
from multiview_detector.utils.nms import nms


def main(args):
    # dataset
    normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    denormalize = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_trans = T.Compose([T.Resize([720, 1280]), T.ToTensor(), normalize, ])
    # change the following directories to fit your needs
    if 'wildtrack_hard' in args.dataset:
        data_path = '/workspace/Data/Wildtrack'
        base = Wildtrack_hard(data_path)
    elif 'wildtrack' in args.dataset:
        data_path = '/workspace/Data/Wildtrack'
        base = Wildtrack(data_path)
    elif 'multiviewx' in args.dataset:
        data_path = '/workspace/Data/MultiviewX'
        base = MultiviewX(data_path)
    else:
        raise Exception('must choose from [wildtrack, multiviewx]')
    test_set = frameDataset(base, train=False, transform=train_trans, grid_reduce=4)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False,
                                              num_workers=2, pin_memory=True)

    if args.model == 'BoosterSHOT':
        model = CA_SA_Detector(test_set, args.arch)
    elif args.model == 'MVDet':
        model = PerspTransDetector(test_set, args.arch)
    else:
        raise Exception('The selected model is not supported.')

    ckpt = torch.load(os.path.join(args.ckpt_dir, 'MultiviewDetector.pth'))
    model.load_state_dict(ckpt)

    os.makedirs(os.path.join(args.ckpt_dir, args.dataset), exist_ok=True)
    res_fpath = os.path.join(args.ckpt_dir, args.dataset, 'test.txt')
    gt_fpath = test_loader.dataset.gt_fpath

    model.eval()
    all_res_list = []
    if res_fpath is not None:
        assert gt_fpath is not None
    for batch_idx, (data, map_gt, imgs_gt, frame) in enumerate(test_loader):
        with torch.no_grad():
            map_res, imgs_res = model(data)
        if res_fpath is not None:
            map_grid_res = map_res.detach().cpu().squeeze()
            v_s = map_grid_res[map_grid_res > args.cls_thres].unsqueeze(1)
            grid_ij = (map_grid_res > args.cls_thres).nonzero()
            if test_loader.dataset.base.indexing == 'xy':
                grid_xy = grid_ij[:, [1, 0]]
            else:
                grid_xy = grid_ij
            all_res_list.append(torch.cat([torch.ones_like(v_s) * frame, grid_xy.float() *
                                            test_loader.dataset.grid_reduce, v_s], dim=1))

        pred = (map_res > args.cls_thres).int().to(map_gt.device)
        true_positive = (pred.eq(map_gt) * pred.eq(1)).sum().item()
        false_positive = pred.sum().item() - true_positive
        false_negative = map_gt.sum().item() - true_positive
        precision = true_positive / (true_positive + false_positive + 1e-4)
        recall = true_positive / (true_positive + false_negative + 1e-4)

    moda = 0
    if res_fpath is not None:
        all_res_list = torch.cat(all_res_list, dim=0)
        res_list = []
        for frame in np.unique(all_res_list[:, 0]):
            res = all_res_list[all_res_list[:, 0] == frame, :]
            positions, scores = res[:, 1:3], res[:, 3]
            ids, count = nms(positions, scores, 20, np.inf)
            res_list.append(torch.cat([torch.ones([count, 1]) * frame, positions[ids[:count], :]], dim=1))
        res_list = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])
        np.savetxt(res_fpath, res_list, '%d')

        recall, precision, moda, modp = 0, 0, 0, 0
        if not args.no_matlab:
            from multiview_detector.evaluation.evaluate import evaluate
            recall, precision, moda, modp = evaluate(os.path.abspath(res_fpath), os.path.abspath(gt_fpath),
                                                        test_loader.dataset.base.__name__)
            print('moda: {:.1f}%, modp: {:.1f}%, precision: {:.1f}%, recall: {:.1f}%'.
                format(moda, modp, precision, recall))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['MVDet', 'BoosterSHOT'])
    parser.add_argument('--cls_thres', type=float, default=0.4)
    parser.add_argument('--arch', type=str, default='resnet18', choices=['vgg11', 'resnet18', 'mobilenet'])
    parser.add_argument('--ckpt_dir')
    parser.add_argument('--dataset', '-d', choices=['wildtrack', 'wildtrack_hard', 'multiviewx'])
    parser.add_argument('--no_matlab', type=int, default=0)

    args = parser.parse_args()

    main(args)