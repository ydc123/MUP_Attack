import os
import torch
import cv2
import numpy as np
import pandas as pd
import scipy.misc
from imageio import imread
from imageio import imsave

class DatasetMetadata_Imagenet():
    def __init__(self, csv):
        self.df = df = pd.read_csv(csv)

    def get_true_label(self, image_id):
        row = self.df[self.df.ImageId == image_id]['TrueLabel']
        assert(len(row) == 1)
        return row.values[0] - 1


def load_images(input_size, img_num, image_dir=None):
    input_dir = '/data1/yangdc/Secondary_Information_Surrogate_Training/imagenet/attack/input/'
    if image_dir == None:
        image_dir = os.path.join(input_dir, 'imagenet_dev')
    images = []
    true_ys = []
    paths = []
    idx = 0
    csv_dir = os.path.join(input_dir, 'dev_dataset.csv')
    datameta = DatasetMetadata_Imagenet(csv_dir)
    filename_list = sorted(os.listdir(image_dir))[0:img_num]
    for fname in filename_list:
        image_id = fname[:-4]
        filepath = os.path.join(image_dir, fname)
        with open(filepath,'rb') as f:
            image = np.array(imread(f))
            image = cv2.resize(image, (input_size[1], input_size[2]))
            image = image.astype(np.float32) / 255
            image = torch.from_numpy(image).cuda()
            image = image.permute(2, 0, 1)[None, :]
        images.append(image)
        paths.append(filepath)
        true_ys.append(datameta.get_true_label(image_id))
        idx += 1
    images = torch.cat(images)
    return paths, images, true_ys