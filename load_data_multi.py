from PIL import Image
import torch
from torch.utils.data import Dataset
import os
from glob import glob
from tqdm import tqdm
import numpy as np

HER2_TYPE_TO_LABEL = {
    '0': 0,
    '1+': 1,
    '2+': 2,
    '3+': 3
}


def read_data(image_path):
    image_list = os.listdir(image_path)
    image_fnames = []
    imageA_path_set = []
    imageB_path_set = []
    image_label_set = []
    for single_img in image_list:  # look for all the subdirs AND the image path
        image_fnames += glob(os.path.join(image_path, single_img))

    print('*** Start loading {} Images ***'.format(len(image_fnames)))

    for image_path in tqdm(image_fnames):
        # a. load image & check if already there
        _, image_name = os.path.split(image_path)
        image_label = HER2_TYPE_TO_LABEL[image_name.split('_')[3][:-4]]
        imageA_path_set.append(image_path)
        imageB_path_set.append(image_path.replace('/HE/', '/IHC/'))
        image_label_set.append(image_label)
    # imageA_path_set, imageB_path_set, image_label_set = MyList(imageA_path_set), MyList(imageB_path_set), MyList(image_label_set)

    return imageA_path_set, imageB_path_set, image_label_set


class MyDataSet_new(Dataset):
    """Patch DataSet"""

    def __init__(self, imageA_path: list, imageB_path: list, images_class: list, transform=None, transformA=None, transformB=None):
        self.imageA_path = imageA_path
        self.imageB_path = imageB_path
        self.images_class = images_class
        self.transform = transform
        self.transformA = transformA
        self.transformB = transformB

    def __len__(self):
        return len(self.imageA_path)

    def __getitem__(self, item):
        img_A = Image.open(self.imageA_path[item])
        img_B = Image.open(self.imageB_path[item])
        label = self.images_class[item]
        if self.transform is not None:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)
        if self.transformA is not None:
            img_A = self.transformA(img_A)
        if self.transformB is not None:
            img_B = self.transformB(img_B)

        return img_A, img_B, label

    @staticmethod
    def collate_fn(batch):
        # [batch_patch, 3, im_w, im_h]
        imagesA, imagesB, labels = tuple(zip(*batch))
        imagesA = torch.stack(imagesA, dim=0)
        imagesB = torch.stack(imagesB, dim=0)
        labels = torch.as_tensor(labels)
        return imagesA, imagesB, labels



