import os, time, sys, math, random, datetime
sys.path.append(os.path.expanduser("~/Documents/omni_research"))
import numpy as np
import cv2, torch, distance
import torch.nn as nn
import omni_torch.visualize.basic as vb
import omni_torch.utils as util
from omni_torch.networks.optimizer.adabound import AdaBound
import researches.img2img.probaV_SR.pvsr_data as data
import researches.img2img.probaV_SR.pvsr_preset as preset
import researches.img2img.probaV_SR.pvsr_model as model
from researches.img2img.probaV_SR.pvsr_augment import *
from researches.img2img.probaV_SR.pvsr_args import parse_arg
import researches.img2img.probaV_SR as init

args = util.get_args(preset.PRESET)
TMPJPG = os.path.expanduser("~/Pictures/tmp.jpg")


def fit(args, net, dataset, optimizer, is_train):
    def avg(list):
        return sum(list) / len(list)
    if is_train:
        net.train()
    else:
        net.eval()
    for batch_idx, (images, targets) in enumerate(dataset):
        print(batch_idx)


def val(args, cfg, net, dataset, optimizer, prior):
    with torch.no_grad():
        fit(args, cfg, net, dataset, optimizer, prior, False)


def main():
    dt = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
    datasets = data.fetch_probaV_data(args, sources=args.train_sources, k_fold=5,
                                         batch_size=args.batch_size_per_gpu, auxiliary_info=[2, 2])
    for idx, (train_set, val_set) in enumerate(datasets):
        loc_loss, conf_loss = [], []
        accuracy, precision, recall, f1_score = [], [], [], []
        print("\n =============== Cross Validation: %s/%s ================ " %
              (idx + 1, len(datasets)))
        net = model.ProbaV_basic(inchannel=args.n_selected_img)
        net = torch.nn.DataParallel(net, device_ids=args.gpu_id, output_device=args.output_gpu_id).cuda()
        torch.backends.cudnn.benchmark = True
        if args.finetune:
            net = util.load_latest_model(args, net, prefix=args.model_prefix_finetune, strict=True)
        optimizer = AdaBound(net.parameters(), lr=args.learning_rate, final_lr=20 * args.learning_rate,
                             weight_decay=args.weight_decay)
        for epoch in range(args.epoch_num):
            psnr_epoch = fit(args, net, train_set, optimizer, is_train=True)
            loc_loss.append(psnr_epoch)
            train_losses = [np.asarray(loc_loss), np.asarray(conf_loss)]
            if val_set is not None:
                accu, pre, rec, f1 = fit(args, net, val_set, optimizer, is_train=False)
                accuracy.append(accu)
                precision.append(pre)
                recall.append(rec)
                f1_score.append(f1)
                val_losses = [np.asarray(accuracy), np.asarray(precision),
                              np.asarray(recall), np.asarray(f1_score)]
            if epoch != 0 and epoch % 10 == 0:
                util.save_model(args, args.curr_epoch, net.state_dict(), prefix=args.model_prefix,
                                keep_latest=20)
            if epoch > 5:
                vb.plot_multi_loss_distribution(
                    multi_line_data=[train_losses, val_losses],
                    multi_line_labels=[["location", "confidence"], ["Accuracy", "Precision", "Recall", "F1-Score"]],
                    save_path=args.loss_log, window=5, name=dt + "cv_%d"%(idx+1),
                    bound=[None, {"low": 0.0, "high": 1.0}],
                    titles=["Train Loss", "Validation Score"]
                )
        # Clean the data for next cross validation
        del net, optimizer
        args.curr_epoch = 0

if __name__ == "__main__":
    main()
