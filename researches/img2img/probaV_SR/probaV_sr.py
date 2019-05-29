import os, time, sys, math, random, datetime
sys.path.append(os.path.expanduser("~/Documents/omni_research"))
import numpy as np
import cv2, torch
import torch.nn as nn
import omni_torch.visualize.basic as vb
import omni_torch.utils as util
from omni_torch.networks.optimizer.adabound import AdaBound
import researches.img2img.probaV_SR.pvsr_data as data
import researches.img2img.probaV_SR.pvsr_preset as preset
import researches.img2img.probaV_SR.pvsr_model as model
from researches.img2img.probaV_SR.pvsr_loss import *
from researches.img2img.probaV_SR.pvsr_args import parse_arg
import researches.img2img.probaV_SR as init

args = util.get_args(preset.PRESET)
TMPJPG = os.path.expanduser("~/Pictures/tmp.jpg")


def fit(args, net, dataset, optimizer, criterion, measure=None, is_train=True):
    def avg(list):
        return sum(list) / len(list)
    if is_train:
        net.train()
    else:
        net.eval()
    aux_loss_importance = 0.5
    epoch_loss, epoch_measure = [], []
    start_time = time.time()
    for batch_idx, (images, blend_target, unblend_target, norm) in enumerate(dataset):
        images, blend_target = images.cuda(), blend_target.cuda()
        prediction, blend_target = net(images, blend_target)

        prediction = [prediction] if type(prediction) not in [list, tuple] else prediction
        blend_target = [blend_target] if type(blend_target) not in [list, tuple] else blend_target

        losses = criterion(prediction, blend_target)
        losses = [losses] if type(losses) not in [list, tuple] else losses
        epoch_loss.append([float(loss.data) for loss in losses])
        if measure:
            basic_measure = measure(prediction, blend_target, losses[0])
            basic_measure = [basic_measure] if type(basic_measure) not in [list, tuple] else basic_measure
            epoch_measure.append([float(loss.data) for loss in basic_measure])
        if is_train:
            loss = sum([l for l in losses])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    if is_train:
        args.curr_epoch += 1
    all_losses = [avg(loss) for loss in list(zip(*epoch_loss))]
    if measure:
        all_measures = [avg(measure) for measure in list(zip(*epoch_measure))]
        print(" --- loss: %.4f, Measure: %.4f, at epoch %04d, cost %.2f seconds ---" %
              (avg(all_losses), avg(all_measures), args.curr_epoch + 1, time.time() - start_time))
        return all_losses, all_measures
    else:
        print(" --- loss: %.4f, at epoch %04d, cost %.2f seconds ---" %
              (avg(epoch_loss), args.curr_epoch + 1, time.time() - start_time))
        return all_losses


def val(args, net, dataset, optimizer, criterion, measure):
    with torch.no_grad():
        return fit(args, net, dataset, optimizer, criterion, measure, False)


def main():
    dt = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
    datasets = data.fetch_probaV_data(args, sources=args.train_sources, k_fold=5,
                                         batch_size=args.batch_size_per_gpu, auxiliary_info=[2, 2])
    for idx, (train_set, val_set) in enumerate(datasets):
        Loss, Measure = [], []
        val_Loss, val_Measure = [], []
        print("\n =============== Cross Validation: %s/%s ================ " %
              (idx + 1, len(datasets)))
        net = model.ProbaV_basic(inchannel=args.n_selected_img)
        net = torch.nn.DataParallel(net, device_ids=args.gpu_id, output_device=args.output_gpu_id).cuda()
        torch.backends.cudnn.benchmark = True
        if args.finetune:
            net = util.load_latest_model(args, net, prefix=args.model_prefix_finetune, strict=True)
        optimizer = AdaBound(net.parameters(), lr=args.learning_rate, final_lr=10 * args.learning_rate,
                             weight_decay=args.weight_decay)
        criterion = ListedLoss(type="l2", reduction="mean")
        #criterion = torch.nn.DataParallel(criterion, device_ids=args.gpu_id, output_device=args.output_gpu_id).cuda()
        measure = MultiMeasure()
        #measure = torch.nn.DataParallel(measure, device_ids=args.gpu_id, output_device=args.output_gpu_id).cuda()
        for epoch in range(args.epoch_num):
            _l, _m = fit(args, net, train_set, optimizer, criterion, measure, is_train=True)
            Loss.append(_l)
            Measure.append(_m)
            if val_set is not None:
                _vl, _vm= val(args, net, val_set, optimizer, criterion, measure)
                val_Loss.append(_vl)
                val_Measure.append(_vm)

            if epoch != 0 and epoch % 10 == 0:
                util.save_model(args, args.curr_epoch, net.state_dict(), prefix=args.model_prefix,
                                keep_latest=20)
            if epoch > 5:
                train_lm = [np.asarray(Loss), np.asarray(Measure)]
                val_lm = [np.asarray(val_Loss), np.asarray(val_Measure)]
                vb.plot_multi_loss_distribution(
                    multi_line_data=[train_lm, val_lm],
                    multi_line_labels=[["location", "confidence"], ["Accuracy", "Precision"]],
                    save_path=args.loss_log, window=5, name=dt + "cv_%d"%(idx+1),
                    bound=[None, {"low": 0.0, "high": 1.0}],
                    titles=["Train Loss", "Validation Score"]
                )
        # Clean the data for next cross validation
        del net, optimizer, criterion, measure
        args.curr_epoch = 0

if __name__ == "__main__":
    main()
