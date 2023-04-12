import torch
import pretrainedmodels
from torch import nn
from tqdm import tqdm
from skimage.io import imsave
import cv2
import numpy as np


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def get_accuracy(output, target, topk=(1, 5)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_image(img, path, img_shape=None):
    if len(img.shape) == 4:
        img = img[0]
    img = img.permute(1, 2, 0).detach().cpu().numpy()
    img = np.clip(img, 0, 255)
    if img_shape != None:
        img = cv2.resize(img, img_shape)
    imsave(path, img.astype(np.uint8))

class ToBGR():
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, image):
        dim = self.dim
        if dim == 0:
            image = image[[2, 1, 0], :, :, :]
        elif dim == 1:
            image = image[:, [2, 1, 0], :, :]
        elif dim == 2:
            image = image[:, :, [2, 1, 0], :, :]
        elif dim == 3:
            image = image[ :, :, :, [2, 1, 0]]
        return image


class Normalize():
    def __init__(self, mean, std, range255):
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)
        self.range255 = range255

    def __call__(self, image):
        dtype = image.type()
        N, C, H, W = image.size()
        if self.range255:
            image = image * 255
        mean = self.mean.clone().type(dtype).view(1, -1, 1, 1)
        std =  self.std.clone().type(dtype).view(1, -1, 1, 1)
        mean = mean.expand_as(image)
        std = std.expand_as(image)
        return image.sub(mean).div(std)

class ModuleList():
    def __init__(self, module_list):
        self.module_list = module_list

    def __call__(self, image):
        x = image
        for module in self.module_list:
            x = module(x)
        return x

    def to_list(self):
        return self.module_list

    def cuda(self):
        pass


def build_preprocess(input_range, input_mean, input_std, range255):
    process = []
    if input_range == 'BGR':
        process.append(ToBGR())
    if input_mean is None:
        input_mean = [0, 0, 0]
    if input_std is None:
        input_std = [1, 1, 1]

    process.append(Normalize(input_mean, input_std, range255))
    return ModuleList(process)

def load_model(args):
    model = pretrainedmodels.__dict__[args.arch](num_classes=1000, pretrained='imagenet').cuda()
    model.eval()
    setting = pretrainedmodels.pretrained_settings[args.arch]['imagenet']
    print(setting)
    input_size = setting['input_size']
    preprocess = build_preprocess(setting['input_space'], setting['mean'], setting['std'], range255=max(setting['input_range'])==255)
    return model, preprocess, input_size


def toTensor(img):
    img = torch.Tensor(img).cuda()
    if len(img.shape) == 3:
        img = img[None, :]
    return img.permute(0, 3, 1, 2)
