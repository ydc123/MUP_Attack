import torch
import torch.nn.functional as F

import numpy as np
import os
from tqdm import tqdm
import argparse
import types
import random

from utils import *
from dataset import load_images

parser = argparse.ArgumentParser()

parser.add_argument('--output', help='output directory', type=str, default='output/expdemo')
parser.add_argument('--img_num', help='number of images to be attacked', type=int, default=1000)
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    help='model architectures')

parser.add_argument('--p', help='pruning percent', type=float, default=0)
parser.add_argument('--type', help='taylor or weight', type=str, default='taylor',
                    choices=['taylor', 'weight', 'grad'])
parser.add_argument('--pruning_mode', help='pruning mode', type=str, default='dynamic',
                    choices=['dynamic', 'none', 'ghost'])

parser.add_argument('--eps', help='epsilon', type=int, default=16)
parser.add_argument('--seed', help='random seed', type=int, default=11037)
parser.add_argument('--iter_num', help='number of iterations for attack.', type=int, default=10)
parser.add_argument('--m_scale', help='m for scale', type=int, default=0)
parser.add_argument('--attack_method', help='attack method', type=str, default='MIFGSM', choices=['MIFGSM', 'SIFGSM', 'TAIGFGSM'])
parser.add_argument('--iter_eps', help='step size of each iteration', type=float, default=2.0)
parser.add_argument('--momentum', help='momentum', type=float, default=1.0)

args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().view(1, -1, 1, 1)
std = torch.Tensor([0.229, 0.224, 0.225]).cuda().view(1, -1, 1, 1)
if 'TAIG' in args.attack_method:
    scale = torch.arange(1, args.m_scale + 1).view(-1, 1, 1, 1).cuda() / args.m_scale
if 'SI' in args.attack_method:
    scale = 0.5 ** torch.arange(args.m_scale).view(-1, 1, 1, 1).cuda()


def set_mask(m, config):
    name = m.__class__.__name__
    if name == 'Conv2d' and m.weight.grad != None:
        p, t = config
        if t == 'taylor':
            norms = m.weight.grad.abs() * m.weight.abs()
        elif t == 'weight':
            norms = m.weight.abs()
        elif t == 'grad':
            norms = m.weight.grad.abs()
        else:
            raise NotImplementedError
        idx = int(norms.numel() * p)
        threshold = norms.view(-1)[norms.view(-1).argsort()[idx]]
        m.mask = (norms >= threshold).float()

def initialize_mask(m):
    name = m.__class__.__name__
    if name == 'Conv2d':
        m.mask = torch.ones(m.weight.shape).cuda()

def reset_mask(m):
    name = m.__class__.__name__
    if name == 'Conv2d':
        m.mask.fill_(1)


def replace(m):
    def new_forward(self, x):
        return F.conv2d(x, self.weight * self.mask, self.bias, self.stride, self.padding, self.dilation, self.groups)

    name = m.__class__.__name__
    if name == 'Conv2d':
        m.forward = types.MethodType(new_forward, m)
def new_inception_v3_forward(self, x):
    prob = args.p # reference to https://github.com/LiYingwei/ghost-network/blob/master/nets/inception_v3.py
    # N x 3 x 299 x 299
    x = self.Conv2d_1a_3x3(x)
    x = F.dropout(x, prob)
    # N x 32 x 149 x 149
    x = self.Conv2d_2a_3x3(x)
    x = F.dropout(x, prob)
    # N x 32 x 147 x 147
    x = self.Conv2d_2b_3x3(x)
    x = F.dropout(x, prob)
    # N x 64 x 147 x 147
    x = self.maxpool1(x)
    x = F.dropout(x, prob)
    # N x 64 x 73 x 73
    x = self.Conv2d_3b_1x1(x)
    x = F.dropout(x, prob)
    # N x 80 x 73 x 73
    x = self.Conv2d_4a_3x3(x)
    x = F.dropout(x, prob)
    # N x 192 x 71 x 71
    x = self.maxpool2(x)
    x = F.dropout(x, prob)
    # N x 192 x 35 x 35
    x = self.Mixed_5b(x)
    x = F.dropout(x, prob)
    # N x 256 x 35 x 35
    x = self.Mixed_5c(x)
    x = F.dropout(x, prob)
    # N x 288 x 35 x 35
    x = self.Mixed_5d(x)
    x = F.dropout(x, prob)
    # N x 288 x 35 x 35
    x = self.Mixed_6a(x)
    x = F.dropout(x, prob)
    # N x 768 x 17 x 17
    x = self.Mixed_6b(x)
    x = F.dropout(x, prob)
    # N x 768 x 17 x 17
    x = self.Mixed_6c(x)
    x = F.dropout(x, prob)
    # N x 768 x 17 x 17
    x = self.Mixed_6d(x)
    x = F.dropout(x, prob)
    # N x 768 x 17 x 17
    x = self.Mixed_6e(x)
    x = F.dropout(x, prob)
    # N x 768 x 17 x 17
    x = self.Mixed_7a(x)
    x = F.dropout(x, prob)
    # N x 1280 x 8 x 8
    x = self.Mixed_7b(x)
    x = F.dropout(x, prob)
    # N x 2048 x 8 x 8
    x = self.Mixed_7c(x)
    x = F.dropout(x, prob)
    # N x 2048 x 8 x 8
    # Adaptive average pooling
    x = self.avgpool(x)
    # N x 2048 x 1 x 1
    x = self.dropout(x)
    # N x 2048 x 1 x 1
    x = torch.flatten(x, 1)
    # N x 2048
    x = self.last_linear(x)
    # N x 1000 (num_classes)
    return x

def new_inception_v4_forword(self, x):
    prob = args.p # reference to https://github.com/LiYingwei/ghost-network/master/nets/inception_v4.py
    for layer in self.features:
        x = layer(x)
        x = F.dropout(x, prob)
    x = self.logits(x)
    return x


def pruning(model, inputs, y, config, preprocess):
    model.apply(reset_mask)
    model.zero_grad()
    inputs = inputs.detach()
    inputs = preprocess(inputs)
    pred = model(inputs)
    loss_func = torch.nn.CrossEntropyLoss()
    loss = loss_func(pred, y)
    loss.backward()
    model.apply(lambda x: set_mask(x, config))

def main():
    config = args.p, args.type
    model, preprocess, input_size = load_model(args)
    if args.pruning_mode == 'dynamic':
        model.apply(initialize_mask)
        model.apply(replace)
    elif args.pruning_mode == 'ghost':
        if args.arch == 'inceptionv3':
            model.forward = types.MethodType(new_inception_v3_forward, model)
        if args.arch == 'inceptionv4':
            model.forward = types.MethodType(new_inception_v4_forword, model)
    input_dir = 'data/images'
    os.makedirs(args.output, exist_ok=True)
    os.system('cp {} {}'.format(os.path.join(input_dir, 'labels*'), os.path.join(args.output)))
    eps = args.eps / 255.0
    alpha = args.iter_eps / 255.0
    paths, images, true_ys = load_images(input_size, args.img_num)
    for idx in tqdm(range(args.img_num)):
        path = paths[idx]
        y = true_ys[idx]
        filename = os.path.basename(path)
        img = images[idx, None]
        y = torch.Tensor([y]).long().cuda()

        min_img = torch.clamp(img - eps, min=0)
        max_img = torch.clamp(img + eps, max=1)
        grad = torch.zeros_like(img)
        momentum = args.momentum
        adv = img.clone().detach().requires_grad_(True)
        loss_func = torch.nn.CrossEntropyLoss()

        for _ in range(args.iter_num):
            if args.pruning_mode == 'dynamic':
                pruning(model, adv, y, config, preprocess)
            adv = adv.detach().requires_grad_(True)
            model.zero_grad()
            if args.m_scale == 0:
                pred = model(preprocess(adv))
                loss = -loss_func(pred, y)
            elif 'TAIG' in args.attack_method: # TAIG
                scaled_inputs = adv * scale
                scaled_inputs = scaled_inputs + 2 * (torch.rand_like(scaled_inputs) - 0.5) * eps
                pred = model(preprocess(scaled_inputs))
                loss = -loss_func(pred, y.repeat(scaled_inputs.shape[0]))
            elif 'SI' in args.attack_method: # SIM
                scaled_inputs = adv * scale
                pred = model(preprocess(scaled_inputs))
                loss = -loss_func(pred, y.repeat(scaled_inputs.shape[0]))
            loss.backward()
            noise = adv.grad.data
            noise /= noise.abs().mean(dim=[1, 2, 3], keepdim=True)
            grad = momentum * grad + noise
            adv = adv - grad.sign() * alpha
            adv = torch.max(adv, min_img)
            adv = torch.min(adv, max_img)
            adv = adv.detach().requires_grad_(True)
        save_image(adv * 255, os.path.join(args.output, filename), (299, 299))


if __name__ == "__main__":
    main()
