import random, cv2
from imgaug import augmenters
import numpy as np
import matplotlib.pyplot as plt

def aug_probaV(bg_color=255):
    aug_list = []
    # Compatible with 16-bit image
    aug_list.append(augmenters.AllChannelsCLAHE(clip_limit=50, tile_grid_size_px=(4, 8)))
    return augmenters.Sequential(aug_list, random_order=False)


def aug_blend_mask(bg_color=255):
    aug_list = []
    aug_list.append(augmenters.GaussianBlur(sigma=(2, 6)))
    return augmenters.Sequential(aug_list, random_order=False)


def combine_mask_with_img(images, args, path, seed, size, device=None):
    aug_seq = aug_probaV()
    aug_seq = aug_seq.to_deterministic()
    blur_seq = aug_blend_mask()
    max_intensity = 2 ** args.img_bit - 1
    # divide by 255, represent normalize the image from 0~1
    #unblended_target = images[args.n_selected_img] * (images[-1] / 255)
    if args.train:
        middle_idx = int((len(images) - 2) / 2)
    else:
        middle_idx = int((len(images)) / 2)
    # get the sum of masked input image, exclude ground truth image
    mask_sum = [np.sum(mask <= 10) for mask in images[middle_idx+1:-1]]
    mask_sum_idx = list(range(middle_idx))
    _, mask_sum_idx = (list(x) for x in zip(*sorted(zip(mask_sum, mask_sum_idx),
                                                    key=lambda pair: pair[0])))
    if len(mask_sum_idx) < args.n_selected_img:
        mask_sum_idx += random.sample(mask_sum_idx, args.n_selected_img - len(mask_sum_idx))
    else:
        mask_sum_idx = mask_sum_idx[:args.n_selected_img]
    # append the ground truth image
    if args.train:
        mask_sum_idx.append(middle_idx)
        middle_idx += 1
    
    blend_mask = [np.clip(blur_seq.augment_image(images[middle_idx+idx]), a_min=0, a_max=255) / 255
                  for idx in mask_sum_idx]
    # The reason we cannot use np.clip is that sometimes the original image pixel intensity
    # may exceed 16384, although they claimed their image is 14-bit
    aug_imgs = [aug_seq.augment_image(np.clip(images[idx] * 4, a_min=0, a_max=max_intensity))
                for idx in mask_sum_idx]
    #aug_imgs = [aug_seq.augment_image(((image - np.min(image)) / (np.max(image) - np.min(image)) * max_intensity)\
                                      #.astype(np.uint16)) for image in images[:args.n_selected_img+1]]
    blended_images = []
    for i, image in enumerate(aug_imgs):
        weight = random.uniform(0.1, 0.4)
        # divide by 4, represent convert 16-bit image back to 14-bit image, but save as 16-bit
        blended_images.append(image * (blend_mask[i] * (1 - weight)) + image * (1 - (blend_mask[i] * weight)))
    # append the unblended ground truth
    if args.train:
        blended_images.append(aug_imgs[-1] * (images[-1] / 255))
    #blended_images = [np.clip(img, a_min=0, a_max=max_intensity) for img in blended_images]
    blended_images = [(img - np.min(img)) / (np.max(img) - np.min(img)) * max_intensity for img in blended_images]
    #save_img(blended_images, blend_mask)
    assert all([np.max(img) < 65536 for img in blended_images])
    return blended_images


#---------------DEBUG CODE----------------
def save_img(imgs, blend_mask, max=65535):
    mask = imgs[-1]
    gt = imgs[-2]
    white = np.zeros((384, 384))
    gt = np.concatenate([mask, white, gt], axis=1)
    blend_mask = np.concatenate(blend_mask[: 9], axis=1) * max
    #blend_mask = np.clip(blend_mask, a_min = 1, a_max = max-1)
    train = np.concatenate([img for i, img in enumerate(imgs[: 9])], axis=1)
    cv2.imwrite("/home/wang/Pictures/tmp.jpg", np.concatenate((train, blend_mask, gt), axis=0) / max * 255)


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
