import os, time, sys, math, random, datetime
sys.path.append(os.path.expanduser("~/Documents/probaV"))
import numpy as np
import cv2, torch
import torch.nn as nn
import omni_torch.visualize.basic as vb
import omni_torch.utils as util
from omni_torch.networks.optimizer.adabound import AdaBound
import researches.img2img.probaV_SR.pvsr_data as data
import researches.img2img.probaV_SR.pvsr_preset as preset
import researches.img2img.probaV_SR.pvsr_model as model
from researches.img2img.probaV_SR.pvsr_args import parse_arguments
from researches.img2img.probaV_SR.pvsr_loss import *

opt = parse_arguments()
args = util.get_args(preset.PRESET, opt=opt)
TMPJPG = os.path.expanduser("~/Pictures/tmp.jpg")


def to_array(lists):
    return [np.asarray(l) for l in list(zip(*lists))]


def fit(args, net, dataset, optimizer, measure=None, is_train=True):
    def avg(list):
        return sum(list) / len(list)
    if is_train:
        net.train()
    else:
        net.eval()
    epoch_loss, epoch_measure = [], []
    start_time = time.time()
    for batch_idx, (images, blend_target, unblend_target, norm) in enumerate(dataset):
        images, blend_target, unblend_target = images.cuda(), blend_target.cuda(), unblend_target.cuda()
        prediction, mae, s_mse = net(images, blend_target)
        mae = torch.sum(mae) / 2
        s_mse = torch.sum(s_mse) / 2

        # Visualize
        if batch_idx == 0 and not is_train:
            # Visualize the image-pairs of the first batch
            for i in range(images.size(0)):
                # enumerate through batch
                img = vb.plot_tensor(args, images[i:i+1, :].permute(1, 0, 2, 3), margin=0)
                pred = vb.plot_tensor(args, prediction[i: i+1, :9], margin=0)
                gt = vb.plot_tensor(args, blend_target[i: i+1, :9], margin=0)
                out = np.concatenate([img, pred, gt], axis=1)
                cv2.imwrite(os.path.join(args.val_log, "epoch_%d_%d.jpg"%(args.curr_epoch, i)), out/ 65536 * 255)

        prediction = [prediction] if type(prediction) not in [list, tuple] else prediction
        blend_target = [blend_target] if type(blend_target) not in [list, tuple] else blend_target

        #losses = criterion(prediction, blend_target)
        #losses = [losses] if type(losses) not in [list, tuple] else losses
        losses = [mae, s_mse]
        epoch_loss.append([float(loss.data) for loss in losses])
        if measure:
            basic_measure = measure(prediction, unblend_target, blend_target)
            basic_measure = [basic_measure] if type(basic_measure) not in [list, tuple] else basic_measure
            epoch_measure.append([float(loss.data) for loss in basic_measure])
        if is_train:
            loss = sum([l for l in losses])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    if is_train:
        print("")
        args.curr_epoch += 1
    all_losses = [avg(loss) for loss in list(zip(*epoch_loss))]
    if measure:
        all_measures = [avg(measure) for measure in list(zip(*epoch_measure))]
        print("epoch: %d --- loss: %.4f, Measure: %.4f, cost %.2f seconds ---" %
              (args.curr_epoch, avg(all_losses), avg(all_measures), time.time() - start_time))
        return all_losses, all_measures
    else:
        print("epoch: %d  --- loss: %.4f, cost %.2f seconds ---" %
              (args.curr_epoch, avg(epoch_loss), time.time() - start_time))
        return all_losses, [0]


def val(args, net, dataset, optimizer, measure):
    with torch.no_grad():
        return fit(args, net, dataset, optimizer, measure, False)
    
    
def test():
    args.train = False
    dataset = data.fetch_probaV_data(args, sources=args.test_sources, shuffle=False,
                                      batch_size=1 / torch.cuda.device_count(), auxiliary_info=[2, 2])[0][0]
    if args.which_model.lower() == "carn":
        net = model.CARN(args.n_selected_img, 64, 3, s_MSE=True, trellis=args.trellis)
    elif args.which_model.lower() == "rdn":
        net = model.RDN(args.n_selected_img, 3, 3, filters=64, s_MSE=True, group=args.n_selected_img,
                        trellis=args.trellis)
    elif args.which_model.lower() == "basic":
        net = model.ProbaV_basic(inchannel=args.n_selected_img)
    else:
        print("args.which_model or -wm should be one of [carn, rdn, basic], "
              "your -wm %s is illegal, and switched to 'basic' automatically" % (args.which_model.lower()))
        net = model.ProbaV_basic(inchannel=args.n_selected_img)
    net = torch.nn.DataParallel(net, device_ids=[0], output_device=0).cuda()
    torch.backends.cudnn.benchmark = True
    net = util.load_latest_model(args, net, prefix=args.model_prefix_finetune, strict=True)
    with torch.no_grad():
        for batch_idx, (images, blend_target, unblend_target, norm) in enumerate(dataset):
            print(batch_idx)
            images, blend_target = images.cuda(), blend_target.cuda(),
            prediction, mae, s_mse = net(images, blend_target, train=False)
            pred = vb.plot_tensor(args, prediction, margin=0)
            cv2.imwrite(os.path.expanduser("~/Pictures/result/%s.jpg"%str(batch_idx).zfill(4)), pred/ 65536 * 255)
            
            


def main():
    dt = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
    datasets = data.fetch_probaV_data(args, sources=args.train_sources, k_fold=args.cross_val, split_val=0.1,
                                         batch_size=args.batch_size_per_gpu, auxiliary_info=[2, 2])
    for idx, (train_set, val_set) in enumerate(datasets):
        Loss, Measure = [], []
        val_Loss, val_Measure = [], []
        print("\n =============== Cross Validation: %s/%s ================ " %
              (idx + 1, len(datasets)))
        if args.which_model.lower() == "carn":
            net = model.CARN(args.n_selected_img, 64, 3, s_MSE=True, trellis=args.trellis)
        elif args.which_model.lower() == "rdn":
            net = model.RDN(args.n_selected_img, 3, 3, filters=64, s_MSE=True, group=args.n_selected_img, trellis=args.trellis)
        elif args.which_model.lower() == "basic":
            net = model.ProbaV_basic(inchannel=args.n_selected_img)
        else:
            print("args.which_model or -wm should be one of [carn, rdn, basic], "
                  "your -wm %s is illegal, and switched to 'basic' automatically"%(args.which_model.lower()))
            net = model.ProbaV_basic(inchannel=args.n_selected_img)
        net = torch.nn.DataParallel(net, device_ids=args.gpu_id, output_device=args.output_gpu_id).cuda()
        torch.backends.cudnn.benchmark = True
        if args.finetune:
            net = util.load_latest_model(args, net, prefix=args.model_prefix_finetune, strict=True)
        optimizer = AdaBound(net.parameters(), lr=args.learning_rate, final_lr=10 * args.learning_rate,
                             weight_decay=args.weight_decay)
        #criterion = ListedLoss(type="l1", reduction="mean")
        #criterion = torch.nn.DataParallel(criterion, device_ids=args.gpu_id, output_device=args.output_gpu_id).cuda()
        measure = MultiMeasure(type="l2", reduction="mean")
        #measure = torch.nn.DataParallel(measure, device_ids=args.gpu_id, output_device=args.output_gpu_id).cuda()
        for epoch in range(args.epoch_num):
            _l, _m = fit(args, net, train_set, optimizer, measure, is_train=True)
            Loss.append(_l)
            Measure.append(_m)
            if val_set is not None:
                _vl, _vm= val(args, net, val_set, optimizer, measure)
                val_Loss.append(_vl)
                val_Measure.append(_vm)

            if (epoch + 1) % 10 == 0:
                util.save_model(args, args.curr_epoch, net.state_dict(), prefix=args.model_prefix,
                                keep_latest=20)
            if (epoch + 1) > 5:
                vb.plot_multi_loss_distribution(
                    multi_line_data=[to_array(Loss) + to_array(val_Loss), to_array(Measure) + to_array(val_Measure)],
                    multi_line_labels=[["train_mae", "train_smse", "val_mae", "val_smse"],
                                       ["train_PSNR", "train_L1", "val_PSNR", "val_L1",]],
                    save_path=args.loss_log, window=3, name=dt + "cv_%d"%(idx+1),
                    bound=[{"low": 0.0, "high": 0.3}, None],
                    titles=["Loss", "Measure"]
                )
        # Clean the data for next cross validation
        del net, optimizer, measure
        args.curr_epoch = 0

if __name__ == "__main__":
    if args.train:
        main()
    if args.test:
        test()
