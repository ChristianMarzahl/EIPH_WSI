from loss.RetinaNetFocalLoss import *

class RetinaNetEIPHFocalLoss(RetinaNetFocalLoss):

    def __init__(self, anchors: Collection[float], gamma: float = 2., alpha: float = 0.25, pad_idx: int = 0,
                 reg_loss_box: LossFunction = F.smooth_l1_loss, reg_loss_score=None):
        super().__init__(anchors, gamma, alpha, pad_idx, reg_loss_box)
        self.metric_names.append('RegImageLoss')
        self.metric_names.append('RegCellLoss')
        self.reg_pred_loss = nn.L1Loss() if reg_loss_score is None else reg_loss_score#nn.MSELoss()#


    def _one_loss(self, clas_pred, bbox_pred, reg_pred, bbox_reg_pred, clas_tgt, bbox_tgt):
        bbox_tgt, clas_tgt = self._unpad(bbox_tgt, clas_tgt)

        reg_prediction_loss = self.reg_pred_loss(clas_tgt.float().mean(), reg_pred) \
            if clas_tgt.size()[0] > 0 else self.reg_pred_loss(tensor(0.).float().to(reg_pred.device), reg_pred)

        matches = match_anchors(self.anchors, bbox_tgt)
        bbox_mask = matches >= 0
        if bbox_mask.sum() != 0:
            bbox_pred = bbox_pred[bbox_mask]
            bbox_tgt = bbox_tgt[matches[bbox_mask]]
            bb_loss = self.reg_loss(bbox_pred, bbox_to_activ(bbox_tgt, self.anchors[bbox_mask]))

            clas_tgt_reg = clas_tgt[matches[bbox_mask]].float()
            clas_tgt_pred = bbox_reg_pred[bbox_mask].view(-1)

            bb_reg_loss = self.reg_pred_loss(clas_tgt_pred, clas_tgt_reg)
        else:
            bb_loss, bb_reg_loss = 0., 0.
        matches.add_(1)
        clas_tgt = clas_tgt + 1
        clas_mask = matches >= 0
        clas_pred = clas_pred[clas_mask]
        clas_tgt = torch.cat([clas_tgt.new_zeros(1).long(), clas_tgt])
        clas_tgt = clas_tgt[matches[clas_mask]]
        return bb_loss, \
               self._focal_loss(clas_pred, clas_tgt) / torch.clamp(bbox_mask.sum(), min=1.), \
               reg_prediction_loss, bb_reg_loss

    def forward(self, output, bbox_tgts, clas_tgts):
        clas_preds, bbox_preds, sizes, regression_preds, bbox_regression_preds = output
        if bbox_tgts.device != self.anchors.device:
            self.anchors = self.anchors.to(clas_preds.device)

        bb_loss = torch.tensor(0, dtype=torch.float32).to(clas_preds.device)
        bb_reg_loss = torch.tensor(0, dtype=torch.float32).to(clas_preds.device)
        focal_loss = torch.tensor(0, dtype=torch.float32).to(clas_preds.device)
        reg_pred_loss = torch.tensor(0, dtype=torch.float32).to(clas_preds.device)
        for cp, bp, rp, brp, ct, bt in zip(clas_preds, bbox_preds, regression_preds, bbox_regression_preds, clas_tgts, bbox_tgts):
            bb, focal, reg_pred, box_reg_pred = self._one_loss(cp, bp, rp, brp, ct, bt)

            bb_loss += bb
            focal_loss += focal
            reg_pred_loss += reg_pred
            bb_reg_loss += box_reg_pred

        self.metrics = dict(zip(self.metric_names, [bb_loss / clas_tgts.size(0), focal_loss / clas_tgts.size(0),
                                                    reg_pred_loss / clas_tgts.size(0), bb_reg_loss / clas_tgts.size(0)]))
        return (bb_loss+focal_loss + focal_loss + reg_pred_loss + bb_reg_loss) / clas_tgts.size(0)