from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io, color
from PIL import Image
import os.path as osp
import os
import torch
import numpy as np
import math
import cv2

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def load_image(path):
    img = Image.open(path).convert('RGB')
    img = transforms.ToTensor()(img)
    # img = io.imread(path)
    # img = img.transpose(2, 0, 1)
    # img = torch.from_numpy(img).float()
    return img

def load_labimage(path):
    img = Image.open(path).convert('RGB')
    img = RGB2LAB()(img)
    img = transforms.ToTensor()(img)
    return img

def save_image(tensor, ori, dir):
    if tensor.size() != ori.size():
        print(tensor.size(), ori.size())
    reclen = tensor.clone() ** 2
    reclen = reclen.sum(dim=0).sqrt()
    orilen = ori.clone() ** 2
    orilen = orilen.sum(dim=0).sqrt()
    tensor = tensor / reclen * orilen

    if tensor.max() > 1:
        tensor = tensor / tensor.max()
    img = tensor.clone().mul(255).clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    io.imsave(dir, img)

def save_labimage(tensor, dir):
    img = tensor.clone().mul(255).clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_Lab2RGB)
    io.imsave(dir, img)

def iter_dir(dir):
    images = []
    for root, dirs, filenames in os.walk(dir):
        for filename in filenames:
            if is_image_file(filename):
                path = osp.join(root, filename)
                images.append(path)
    return images

class RGB2LAB():
    def __call__(self, img):
        npimg = np.array(img)
        img = cv2.cvtColor(npimg, cv2.COLOR_RGB2Lab)
        pilimg = Image.fromarray(img)
        return pilimg

# class LAB2Tensor():
#     def __call__(self, labimg):
#         labimg = labimg.astype(np.float) / 127
#         return torch.from_numpy(labimg.transpose(2, 0, 1)).float()

class ColorPerturb():

    def __call__(self, pilimg):
        img = np.array(pilimg).astype(np.float)

        if np.random.uniform(-2, 1) > 0:
            perturb = np.random.uniform(0.6, 1.4)
            img[:, :, 0] *= perturb

            perturb = np.random.uniform(0.6, 1.4)
            img[:, :, 2] *= perturb
        else:
            tmin = np.random.uniform(0.4, 0.6)
            tmax = np.random.uniform(1.4, 1.6)
            w, h, c= img.shape

            if np.random.uniform(-1, 1) > 0:
                dim = h
            else:
                dim = w
            steps = (tmax - tmin) / dim
            temp = np.arange(tmin, tmax, steps)
            if len(temp) == dim - 1:
                temp = np.concatenate([temp, [tmax]])
            elif len(temp) == dim + 1:
                temp = temp[:-1]
            if np.random.uniform(-1, 1) > 0:
                temp = temp[::-1]

            perturb = np.diag(temp)
            if dim == h:
                img[:, :, 0] = np.dot(img[:, :, 0], perturb)
                img[:, :, 2] = np.dot(img[:, :, 2], perturb)
            else:
                img[:, :, 0] = np.dot(perturb, img[:, :, 0])
                img[:, :, 2] = np.dot(perturb, img[:, :, 2])

        if img.max() > 255:
            img = img / img.max() * 255
        img = img.astype('uint8')
        img = Image.fromarray(img)
        return img

class FlatImageFolder(Dataset):
    def __init__(self, root, transform=None, pert_transform=None):
        self.imgs = iter_dir(root)
        if len(self.imgs) == 0:
            raise (RuntimeError("Found 0 images in folders of: " + root + "\n"))
        self.root = root
        self.transform = transform
        self.pert_transform = pert_transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        path = self.imgs[idx]
        img = Image.open(path).convert('RGB')
        perturb = img.copy()
        if self.pert_transform:
            perturb = self.pert_transform(perturb)

        if self.transform is not None:
            img = self.transform(img)
            perturb = self.transform(perturb)

        return perturb, img