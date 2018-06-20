import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, dataset
import numpy as np


import utils
from transformer_net import TransformerNet

from skimage import io
import argparse
import os
import os.path as osp
import sys
import time
from datetime import datetime


def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: x.mul(255))
    ])
    pert_transform = transforms.Compose([
        utils.ColorPerturb()
    ])
    trainset = utils.FlatImageFolder(args.dataset, transform, pert_transform)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    model = TransformerNet()
    if args.gpus is not None:
        model = nn.DataParallel(model, device_ids=args.gpus)
    else:
        model = nn.DataParallel(model)
    if args.resume:
        state_dict = torch.load(args.resume)

        model.load_state_dict(state_dict)

    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    criterion = nn.MSELoss()

    start_time = datetime.now()

    for e in range(args.epochs):
        model.train()
        count = 0
        acc_loss = 0.0
        for batchi, (pert_img, ori_img) in enumerate(trainloader):
            count += len(pert_img)
            if args.cuda:
                pert_img = pert_img.cuda(non_blocking=True)
                ori_img = ori_img.cuda(non_blocking=True)

            optimizer.zero_grad()

            rec_img = model(pert_img)
            loss = criterion(rec_img, ori_img)
            loss.backward()
            optimizer.step()

            acc_loss += loss.item()

            if (batchi + 1) % args.log_interval == 0:
                mesg = '{}\tEpoch {}: [{}/{}]\ttotal loss: {:.6f}'.format(
                    time.ctime(), e + 1, count, len(trainset), acc_loss/(batchi + 1))
                print(mesg)

        if args.checkpoint_dir:
            model.eval().cpu()
            ckpt_filename = 'ckpt_epoch_' + str(e+1) + '.pth'
            ckpt_path = osp.join(args.checkpoint_dir, ckpt_filename)
            torch.save(model.state_dict(), ckpt_path)
            model.cuda().train()
            print('Checkpoint model at epoch %d saved' % (e+1))

    model.eval().cpu()
    if args.save_model_name:
        model_filename = args.save_model_name
    else:
        model_filename = "epoch_" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_') + ".model"
    model_path = osp.join(args.save_model_dir, model_filename)
    torch.save(model.state_dict(), model_path)

    end_time = datetime.now()

    print('Finished training after %s, trained model saved at %s' % (end_time - start_time , model_path))

def check_path(path):
    try:
        if not osp.exists(path):
            os.makedirs(path)
    except OSError as e:
        print(e)
        sys.exit(1)

def evaluate(args):
    # device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    model = TransformerNet()
    state_dict = torch.load(args.model)

    if args.gpus is not None:
        model = nn.DataParallel(model, device_ids=args.gpus)
    else:
        model = nn.DataParallel(model)
    model.load_state_dict(state_dict)
    if args.cuda:
        model.cuda()

    with torch.no_grad():
        for root, dirs, filenames in os.walk(args.input_dir):
            for filename in filenames:
                if utils.is_image_file(filename):
                    impath = osp.join(root, filename)
                    img = utils.load_image(impath)
                    img = img.unsqueeze(0)
                    if args.cuda:
                        img.cuda()
                    rec_img = model(img).cpu()
                    save_path = osp.join(args.output_dir, filename)
                    utils.save_image(rec_img[0], img[0].cpu(), save_path)



def main():
    main_arg_parser = argparse.ArgumentParser()
    subparsers = main_arg_parser.add_subparsers(title='subcommands', dest='subcommand')

    train_parser = subparsers.add_parser('train', help='train the network')
    train_parser.add_argument('--epochs', type=int, default=2, help='number of training epochs, default is 2')
    train_parser.add_argument('--batch-size', type=int, default=30, help='training batch size, default is 4')
    train_parser.add_argument('--dataset', required=True, help='path to training dataset, the path should '
                                'point to a folder containing another folder with all the training images')
    train_parser.add_argument('--save-model-dir', default='model', help='directory of the model to be saved')
    train_parser.add_argument('--save-model-name', default=None, help='save model name')
    train_parser.add_argument('--image-size', type=int, default=256, help='size of training images, default is 256')
    train_parser.add_argument('--cuda', action='store_true', default=False, help='run on GPU')
    train_parser.add_argument('--seed', type=int, default=42, help='random seed for training')
    train_parser.add_argument('--lr', type=float, default=1e-3, help='learning rate, default is 0.001')
    train_parser.add_argument('--log-interval', type=int, default=100, help='number of images after which the training loss is logged,'
                                                                            ' default is 500')
    train_parser.add_argument('--checkpoint-dir', default=None, help='checkpoint model saving directory')
    train_parser.add_argument('--resume', default=None, help='resume training from saved model')
    train_parser.add_argument('--gpus', type=int, nargs='*', default=None, help='specify GPUs to use')

    eval_parser = subparsers.add_parser('eval', help='eval the network')
    eval_parser.add_argument('--input-dir', required=True, help='path to input image directory')
    eval_parser.add_argument('--output-dir', default='output', help='path to output image directory')
    eval_parser.add_argument('--model', required=True, help='saved model to be used for evaluation')
    eval_parser.add_argument('--cuda', action='store_true', default=False, help='run on GPU')
    eval_parser.add_argument('--gpus', type=int, nargs='*', default=None, help='specify GPUs to use')
    args = main_arg_parser.parse_args()

    if args.subcommand is None:
        print('ERROR: specify either train or eval')
        sys.exit(1)
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)
    if args.subcommand == 'train':
        check_path(args.save_model_dir)
        if args.checkpoint_dir:
            check_path(args.checkpoint_dir)
        train(args)
    else:
        check_path(args.output_dir)
        evaluate(args)


if __name__ == '__main__':
    main()