import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import sys
import shutil
from distutils.dir_util import copy_tree
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from multiview_detector.dataset.wildtrack_bbox import WildtrackBBOX
from multiview_detector.model.bbox_classifier import BBOXClassifier
from multiview_detector.utils.logger import Logger
from multiview_detector.utils.draw_curve import draw_curve
from multiview_detector.trainer import BBOXTrainer
from multiview_detector.evaluation.evaluate import matlab_eval


def main():
    # settings
    parser = argparse.ArgumentParser(description='Multiview detector')
    parser.add_argument('--reID', action='store_true')
    parser.add_argument('--cls_thres', type=float, default=0.4)
    parser.add_argument('--soften', type=float, default=1, help='soften coefficient for softmax')
    parser.add_argument('--arch', type=str, default='vgg11')
    parser.add_argument('-d', '--dataset', type=str, default='wildtrack_bbox', choices=['wildtrack_bbox'])
    parser.add_argument('--test_type', type=str, default='test', choices=['val', 'test'])
    parser.add_argument('-j', '--num_workers', type=int, default=4)
    parser.add_argument('-b', '--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--epochs', type=int, default=60, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-1, metavar='LR', help='learning rate (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--seed', type=int, default=None, help='random seed (default: None)')
    args = parser.parse_args()

    # seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    # dataset
    if 'wildtrack' in args.dataset:
        data_path = os.path.expanduser('~/Data/wildtrack_bbox')
        normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        train_trans = T.Compose([T.Resize([256, 128]), T.ToTensor(), normalize, ])
        test_trans = T.Compose([T.Resize([256, 128]), T.ToTensor(), normalize, ])
        train_set = WildtrackBBOX(data_path, split='train', transform=train_trans)
        val_set = WildtrackBBOX(data_path, split='val', transform=test_trans)  # ,train_ratio=0.9975
        test_set = WildtrackBBOX(data_path, split=args.test_type, transform=test_trans)  # ,train_ratio=0.9975
    else:
        raise Exception

    # network specific setting

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.num_workers, pin_memory=True)

    logdir = f'logs/{args.dataset}/' + datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S') \
        if not args.resume else f'logs/{args.dataset}/{args.resume}'
    if args.resume is None:
        os.makedirs(logdir, exist_ok=True)
        copy_tree('./multiview_detector', logdir + '/scripts/multiview_detector')
        for script in os.listdir('.'):
            if script.split('.')[-1] == 'py':
                dst_file = os.path.join(logdir, 'scripts', os.path.basename(script))
                shutil.copyfile(script, dst_file)
        sys.stdout = Logger(os.path.join(logdir, 'log.txt'), )
    print('Settings:')
    print(vars(args))

    # model
    model = BBOXClassifier(train_set.num_cam, args.arch).cuda()
    # model = nn.DataParallel(model)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 20, 1)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
    #                                                 steps_per_epoch=len(train_loader), epochs=args.epochs)

    # loss
    criterion = nn.CrossEntropyLoss().cuda()

    # draw curve
    x_epoch = []
    train_loss_s = []
    train_prec_s = []
    og_test_loss_s = []
    og_test_prec_s = []

    trainer = BBOXTrainer(model, criterion, args.cls_thres)

    # learn
    if args.resume is None:
        # print('Testing...')
        # trainer.test(test_loader)

        for epoch in range(1, args.epochs + 1):
            print('Training...')
            train_loss, train_prec = trainer.train(epoch, train_loader, optimizer, args.log_interval, scheduler)
            print('Testing...')
            test_loss, test_prec = trainer.test(val_loader)

            x_epoch.append(epoch)
            train_loss_s.append(train_loss)
            train_prec_s.append(train_prec)
            og_test_loss_s.append(test_loss)
            og_test_prec_s.append(test_prec)
            draw_curve(os.path.join(logdir, 'learning_curve.jpg'), x_epoch, train_loss_s, train_prec_s,
                       og_test_loss_s, og_test_prec_s)
        # save
        torch.save(model.state_dict(), os.path.join(logdir, 'MultiviewDetector.pth'))
    else:
        resume_dir = f'logs/{args.dataset}/' + args.resume
        resume_fname = resume_dir + '/MultiviewDetector.pth'
        model.load_state_dict(torch.load(resume_fname))
        model.eval()
    print('Test loaded model...')
    trainer.test(test_loader, res_fpath=os.path.join(logdir, f'{args.test_type}.txt'))
    matlab_eval(os.path.abspath(os.path.join(logdir, f'{args.test_type}.txt')), os.path.abspath(test_set.gt_fpath))
    pass


if __name__ == '__main__':
    main()
