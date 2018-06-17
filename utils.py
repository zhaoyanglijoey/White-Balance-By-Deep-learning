from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io
from PIL import Image
import os.path as osp
import os
import torch
import numpy as np


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


def save_image(tensor, dir):
    img = tensor.clone().mul(255).clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    io.imsave(dir, img)


def iter_dir(dir):
    images = []
    for root, dirs, filenames in os.walk(dir):
        for filename in filenames:
            if is_image_file(filename):
                path = osp.join(root, filename)
                images.append(path)
    return images


class ColorPerturb():

    def __call__(self, tensor_img):
        img = tensor_img.numpy()


        if np.random.uniform(-2, 1) > 0:
            perturb = np.random.uniform(0.6, 1.4)
            img[0, :, :] *= perturb

            perturb = np.random.uniform(0.6, 1.4)
            img[2, :, :] *= perturb
        else:
            tmin = np.random.uniform(0.4, 0.6)
            tmax = np.random.uniform(1.4, 1.6)
            c, w, h= img.shape

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
                img[0, :, :] = np.dot(img[0, :, :], perturb)
                img[2, :, :] = np.dot(img[2, :, :], perturb)
            else:
                img[0, :, :] = np.dot(perturb, img[0, :, :])
                img[2, :, :] = np.dot(perturb, img[2, :, :])

        if (img > 1).sum() > 0:
            maxV = img.max()
            img = img / maxV
        img = torch.from_numpy(img)
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
        if self.transform is not None:
            img = self.transform(img)

        perturb = img.clone()

        if self.pert_transform:
            perturb = self.pert_transform(perturb)

        return perturb, img