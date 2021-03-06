def GeneralPattern_01(args):
    args.path = "~/Pictures/dataset/SR"
    args.code_name = "probaV_exp"
    args.cover_exist = True
    args.train = True

    args.deterministic_train = False
    args.learning_rate = 1e-4
    args.epoches_per_phase = 1
    args.epoch_num = 100
    args.cross_val = 1
    
    args.output_gpu_id = 0
    args.random_order_load = False
    args.batch_size_per_gpu = 4
    args.loading_threads = 2
    args.img_channel = 1
    args.curr_epoch = 0
    
    args.do_imgaug = True
    # original image is claimed to be 14-bit, but it will be
    # converted to 16-bit during loading
    args.img_bit = 16
    args.normalize_img = False
    args.img_bias = (0.5, 0.5, 0.5)

    args.finetune = False
    return args


def UniquePattern_01(args):
    args.which_model="basic"
    args.trellis = False
    args.model_prefix = "sr"
    args.model_prefix_finetune = "sr"
    args.train_sources = ["probaV/train"]#, "probaV/train/RED"]
    args.test_sources = ["probaV/test"]  # , "probaV/train/RED"]
    # Select n images from each folder for train, val and test
    # if this is set to 0 then it means select all of them
    # should not be larger than 20 (20 is the maximum number)
    args.n_selected_img = 9
    args.filters = 64
    args.s_MSE = False
    return args


def RuntimePattern(args):
    args.train = False
    args.test = False
    args.half_precision = False
    return args


PRESET = {
    "general": GeneralPattern_01,
    "unique": UniquePattern_01,
    "runtime": RuntimePattern,
}