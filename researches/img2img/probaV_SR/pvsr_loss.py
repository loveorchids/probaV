import torch
import torch.nn as nn

class ListedLoss(nn.Module):
    def __init__(self, type, reduction="sum"):
        super().__init__()
        if type == "l2":
            self.loss_func = nn.MSELoss(reduction=reduction)
        elif type == "l1":
            self.loss_func = nn.L1Loss(reduction=reduction)
        else:
            raise NotImplementedError

    def forward(self, pred_list, target_list, reduction=False):
        pred_list = [pred_list] if type(pred_list) is not list else pred_list
        target_list = [target_list] if type(target_list) is not list else target_list
        assert len(pred_list) == len(target_list)
        loss = 0
        for i, pred in enumerate(pred_list):
            if reduction:
                loss += self.loss_func(pred, target_list[i]) / pred.nelement()
            else:
                loss += self.loss_func(pred, target_list[i])
        return loss


class MixedLoss(nn.Module):
    """
    Implemented aaccording to:
    https://kelvins.esa.int/proba-v-super-resolution/scoring/
    """
    def __init__(self):
        super().__init__()
        self.mse = ListedLoss(type="l2", reduction="sum")

    def forward(self, pred, unblended_y, blended_y):
        MSE = self.mse(pred, blended_y, reduction=True)
        # when pred is a list, the first element represent the image
        # while the rest elements represent the evaluator's feature map
        SR = pred[0][:, :, 3: 381, 3: 381]
        cMSEs = []
        cMSE, min_coord = torch.tensor(999999999.9), [0, 0]
        for u in range(0, 7):
            for v in range(0, 7):
                HR_uv = unblended_y[0][:, :, u: u + 378, v: v + 378]
                # mask pixel is less than this value
                # because we normalize the image to -1 ~ 1
                co_eff = 1 / torch.sum(HR_uv < -0.995).float()
                b = co_eff * torch.sum(HR_uv - SR)
                #cmse = torch.sum((HR_uv - (SR + b)) ** 2)
                cmse = co_eff * self.mse(HR_uv, SR + b)
                print(cmse)
                if float(cmse) < float(cMSE):
                    cMSE = cmse
                    min_coord = [u, v]
        return MSE, cMSE, min_coord


class MultiMeasure(nn.Module):
    def __init__(self, type="l1", reduction="mean", half_precision=False):
        super().__init__()
        self.mae = ListedLoss(type=type, reduction=reduction)
        self.mse = nn.MSELoss(reduction="sum")
        self.half_precision = half_precision

    def forward(self, pred, target, blended_target):
        """
        cPSNR Implemented aaccording to:
        https://kelvins.esa.int/proba-v-super-resolution/scoring/
        """
        SR = pred[0][:, :, 3: 381, 3: 381]
        cMSEs = []
        cMSE, min_coord = torch.tensor(999999999.9), [0, 0]
        for u in range(0, 7):
            for v in range(0, 7):
                HR_uv = target[:, :, u: u + 378, v: v + 378]
                # mask pixel is less than this value
                # because we normalize the image to -1 ~ 1
                if self.half_precision:
                    co_eff = 1 / torch.sum(HR_uv > 5).float()
                    b = co_eff * torch.sum(((HR_uv - SR) * (HR_uv > 5).half()).float())
                else:
                    co_eff = 1 / torch.sum(HR_uv > 5).float()
                    b = co_eff * torch.sum((HR_uv - SR) * (HR_uv > 5).float())
                # cmse = torch.sum((HR_uv - (SR + b)) ** 2)
                cmse = co_eff * self.mse(HR_uv, SR + b)
                #print(cmse)
                if float(cmse) < float(cMSE):
                    cMSE = cmse
                    coord = [u, v]
        cPSNR = -10 * torch.log10(cMSE)
        try:
            MAE = self.mae(SR, target[:, :, coord[0]: coord[0] + 378, coord[1]: coord[1] + 378])
        except NameError:
            return cPSNR, self.mae(pred, target)
        return  cPSNR, MAE