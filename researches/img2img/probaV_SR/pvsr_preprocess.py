from imgaug import augmenters
import numpy as np
import matplotlib.pyplot as plt

def aug_probaV(bg_color=255):
    aug_list = []
    # Compatible with 16-bit image
    aug_list.append(augmenters.AllChannelsCLAHE(clip_limit=(10, 80), tile_grid_size_px=(4, 8)))
    return augmenters.Sequential(aug_list, random_order=False)


def aug_blend_mask(bg_color=255):
    aug_list = []
    aug_list.append(augmenters.GaussianBlur(sigma=(2, 6)))
    return augmenters.Sequential(aug_list, random_order=False)

def combine_mask_with_img(images, args, path, seed, size, device=None):
    aug_seq = aug_probaV()
    aug_seq = aug_seq.to_deterministic()
    blur_seq = aug_blend_mask()
    # divide by 255, represent normalize the image from 0~1
    masks = [blur_seq.augment_image(mask) / 255 for mask in images[args.n_selected_img + 1:]]
    # multiply by 4, represent convert 14-bit image to 16-bit image
    images = [aug_seq.augment_image(image * 4) for image in images[:args.n_selected_img+1]]
    blended_images = []
    for i, image in enumerate(images):
        # divide by 4, represent convert 16-bit image back to 14-bit image, but save as 16-bit
        blended_images.append((image * masks[i] + image * (1 - (masks[i] * 0.1))).astype(np.uint16))
    return blended_images

def visualize(blended_images):
    fig, ax = plt.subplots(nrows=len(blended_images), ncols=1,
                           figsize=(2, 2*len(blended_images)))
    for i, image in enumerate(blended_images):
        ax[i].imshow(image, cmap="gray")
    plt.show()