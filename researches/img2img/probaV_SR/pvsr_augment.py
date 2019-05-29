from imgaug import augmenters


def aug_probaV(bg_color=255):
    aug_list = []
    # Compatible with 16-bit image
    aug_list.append(augmenters.AllChannelsCLAHE(clip_limit=(10, 80), tile_grid_size_px=(4, 8)))

    # aug_list.append(augmenters.normalize_images())
    return augmenters.Sequential(aug_list, random_order=False)


def aug_blend_mask(bg_color=255):
    aug_list = []
    # OK to use
    aug_list.append(augmenters.GaussianBlur(sigma=(2, 6)))

    # aug_list.append(augmenters.normalize_images())
    return augmenters.Sequential(aug_list, random_order=False)