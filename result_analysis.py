import network.utils as utils
import sys
import os
import os.path as osp
import collections
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from skimage import io, color

def normalize(array, dim):
    norm = np.sqrt((array ** 2).sum(dim))
    norm[norm==0] = 1
    norm = np.expand_dims(norm, dim)
    array = array / norm
    return array


def main():
    ori_dir = sys.argv[1]
    rec_dir = sys.argv[2]
    count = 0
    mean_ang_dif = 0
    total_ang_dif = 0
    for root, dirs, filenames in os.walk(rec_dir):
        count += len(filenames)
        for filename in filenames:
            if not (filename.endswith(".jpg") or filename.endswith('.png') or filename.endswith(".jpeg")):
                continue
            rec_img = io.imread(osp.join(root, filename)).astype(np.float)
            ori_img = io.imread(osp.join(ori_dir, filename)).astype(np.float)
            mask = ori_img.sum(2) > 150
            rec_img = normalize(rec_img, 2)
            ori_img = normalize(ori_img, 2)
            ang_dif = np.rad2deg(np.arccos(np.mean((rec_img[mask] * ori_img[mask]).sum(1))))
            total_ang_dif += ang_dif

    mean_ang_dif = total_ang_dif / count
    print('mean angular difference between %s and %s: %f' %(rec_dir, ori_dir, mean_ang_dif))


if __name__ == '__main__':
    main()