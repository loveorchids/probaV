import random
from imgaug import augmenters
import numpy as np
import matplotlib.pyplot as plt

def aug_probaV(bg_color=255):
    aug_list = []
    # Compatible with 16-bit image
    aug_list.append(augmenters.AllChannelsCLAHE(clip_limit=(10, 20), tile_grid_size_px=(4, 8)))
    return augmenters.Sequential(aug_list, random_order=False)


def aug_blend_mask(bg_color=255):
    aug_list = []
    aug_list.append(augmenters.GaussianBlur(sigma=(2, 6)))
    return augmenters.Sequential(aug_list, random_order=False)


def combine_mask_with_img(images, args, path, seed, size, device=None):
    aug_seq = aug_probaV()
    aug_seq = aug_seq.to_deterministic()
    blur_seq = aug_blend_mask()
    max_intensity = 2 ** args.img_bit
    #images = [((img - np.min(img)) / (np.max(img) - np.min(img)) * 16384).astype(np.uint16)
              #for img in images]
    # divide by 255, represent normalize the image from 0~1, and convert to 16-bit img by multiplying 4
    unblended_target = images[args.n_selected_img] * (images[-1] / 255)#).astype(np.uint16)
    blend_mask = [blur_seq.augment_image(mask) / 255 for mask in images[args.n_selected_img + 1:]]
    # convert 14-bit image to 16-bit image by mulyiplying 4
    aug_imgs = [aug_seq.augment_image(image * 4) for image in images[:args.n_selected_img+1]]
    blended_images = []
    for i, image in enumerate(aug_imgs):
        weight = min(max(random.gauss(0.15, 0.25), 0.05), 0.5)
        # divide by 4, represent convert 16-bit image back to 14-bit image, but save as 16-bit
        blended_images.append((image * blend_mask[i] + image * (1 - (blend_mask[i] * weight))).astype(np.uint16))
    # append the unblended ground truth
    blended_images.append(unblended_target)
    blended_images = [(img - np.min(img)) / (np.max(img) - np.min(img)) * max_intensity
                      for img in blended_images]
    return blended_images

def visualize(blended_images):
    fig, ax = plt.subplots(nrows=len(blended_images), ncols=1,
                           figsize=(2, 2*len(blended_images)))
    for i, image in enumerate(blended_images):
        ax[i].imshow(image, cmap="gray")
    plt.show()

if __name__ == "__main__":
    class Bunch(object):
        def __init__(self, adict):
            self.__dict__.update(adict)
    import cv2, os, glob
    folder = os.path.expanduser("~/Pictures/dataset/SR/probaV/train/NIR/imgset0594")
    HR = os.path.join(folder, "HR.png")
    SM = os.path.join(folder, "SM.png")
    LR = sorted(glob.glob(os.path.join(folder, "LR*")))
    QM = sorted(glob.glob(os.path.join(folder, "QM*")))
    datapath = LR + [HR] + QM + [SM]
    images = [cv2.imread(path, -1) for path in datapath]
    args = Bunch({"n_selected_img": 21, "img_bit": 16})
    combine_mask_with_img(images, args, 0, 0, 0)
