import os, glob, random
import numpy as np
import torch, cv2
from torch.utils.data import *
from omni_torch.data.arbitrary_dataset import Arbitrary_Dataset
from researches.img2img.probaV_SR.pvsr_preprocess import *
import omni_torch.data.data_loader as omth_loader
#from omni_torch.data.path_loader import *
import omni_torch.utils as util


def load_path_from_folder(args, length, paths, dig_level=0):
    def load_path(args, path, dig_level):
        current_folders = [path]
        # Do not delete the following line, we need this when dig_level is 0.
        sub_folders = []
        while dig_level > 0:
            sub_folders = []
            for sub_path in current_folders:
                sub_folders += glob.glob(sub_path + "/*")
            current_folders = sub_folders
            dig_level -= 1
        sub_folders = []
        for _ in current_folders:
            # Low resolution image
            LR = sorted(glob.glob(_ + "/LR*.png"))
            # Mask for low resolution image
            QM = sorted(glob.glob(_ + "/QM*.png"))
            assert len(LR) == len(QM)
            HR = os.path.join(_, "HR.png")
            SM = os.path.join(_, "SM.png")
            sub_folders.append([LR, QM, HR, SM])
        return sub_folders
    output = []
    if type(paths) is str:
        paths = [paths]
    for path in paths:
        sub_folders = load_path(args, path, dig_level)
        output += sub_folders
    output.sort()
    return [output]


def selective_image_loading(args, items, seed, size, pre_process=None, rand_aug=None,
               bbox_loader=None):
    LR, QM, HR, SM = items
    if args.n_selected_img == -1:
        # Randomly select n images from LR images
        num_select = random.sample(list(range(len(LR))), random.randint(0, len(LR)))
        selected_LR = [LR[num] for num in num_select]
        selected_QM = [QM[num] for num in num_select]
    elif args.n_selected_img == 0:
        # Select all images from LR images
        selected_LR = LR
        selected_QM = QM
    else:
        # Select n images from LR images
        # 20 is the maximum LR images in the dataset
        assert 0 < args.n_selected_img <= 20
        n_select = args.n_selected_img
        num_select = []
        while n_select > len(LR):
            num_select += list(range(len(LR)))
            n_select -= len(LR)
        num_select += random.sample(list(range(len(LR))), n_select)
        selected_LR = [LR[num] for num in num_select]
        selected_QM = [QM[num] for num in num_select]
    datapath = selected_LR + [HR] + selected_QM + [SM]
    """
    input_order = []
    for i in range(args.n_selected_img):
        input_order.append(i)
        input_order.append(i + args.n_selected_img + 1)
    input_order += [args.n_selected_img, 2 * args.n_selected_img + 1]
    datapath = [datapath[i] for i in input_order]
    """
    images = omth_loader.read_image(args, datapath, seed, size, pre_process, rand_aug)
    # Transfer the 8-bit to 14-bit image
    #for i, image in enumerate(images[args.n_selected_img + 1 :]):
        # 64=2^6 is the number to transfer from 8-bit to 14-bit
        #images[args.n_selected_img + 1 + i] =  images[args.n_selected_img + 1 + i].astype(np.uint16) * 64
    #input_tensor = torch.cat([omth_loader.to_tensor(args, images[i]) for i in input_order])
    #output_tensor = torch.cat([omth_loader.to_tensor(args, images[i]) for i in output_order])
    # If args.n_selected_img is 0 or -1, for each batch, the first dimension of
    # input_tensor is not fixed. Thus we need to use sr_collector to combine
    # each bach, instead of using PyTorch's default batch collector.
    return torch.cat(images[:-1]), images[-1]


def sr_collector(batch):
    imgs, labels = [], []
    for sample in batch:
        imgs.append(sample[0][0])
        labels.append(sample[0][1])
    imgs = torch.stack(imgs, 0)
    labels = torch.stack(labels, 0)
    return imgs, labels


def fetch_probaV_data(args, sources, auxiliary_info, batch_size, batch_size_val=None,
                         shuffle=True, split_val=0.0, k_fold=1, pre_process=None, aug=None):
    args.loading_threads = round(args.loading_threads * torch.cuda.device_count())
    batch_size = batch_size * torch.cuda.device_count()
    if batch_size_val is None:
        batch_size_val = batch_size
    else:
        batch_size_val *= torch.cuda.device_count()
    dataset = []
    for i, source in enumerate(sources):
        subset = Arbitrary_Dataset(args, sources=[source], step_1=[load_path_from_folder],
                                   step_2=[selective_image_loading], auxiliary_info=[auxiliary_info[i]],
                                   pre_process=[combine_mask_with_img], augmentation=[aug])
        subset.prepare()
        dataset.append(subset)

    if k_fold > 1:
        return util.k_fold_cross_validation(args, dataset, batch_size, batch_size_val, k_fold,
                                            collate_fn=sr_collector)
    else:
        if split_val > 0:
            return util.split_train_val_dataset(args, dataset, batch_size, batch_size_val, split_val,
                                                collate_fn=sr_collector)
        else:
            kwargs = {'num_workers': args.loading_threads, 'pin_memory': True}
            train_set = DataLoader(ConcatDataset(dataset), batch_size=batch_size,
                                   shuffle=shuffle, collate_fn=sr_collector, **kwargs)
            return [(train_set, None)]


if __name__ == "__main__":
    pass
