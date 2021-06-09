# -*- coding: utf-8 -*-
# --------------------------------------------------------
# RefineDet in PyTorch
# Written by Dongdong Wang
# Official and original Caffe implementation is at
# https://github.com/sfzhang15/RefineDet
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import match, match_with_flags, log_sum_exp


class CPDLoss(nn.Module):
    """
    """

    def __init__(self, overlap_thresh, neg_pos_ratio, variance):
        super(CPDLoss, self).__init__()
        self.overlap_thresh = overlap_thresh
        self.num_classes = 2
        self.neg_pos_ratio = neg_pos_ratio
        self.variance = variance

    def forward(self, predictions, anchors, targets):
        """Binary box and classification Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and anchors boxes from SSD net.
                conf shape: torch.size(batch_size, num_anchors, 2)
                loc shape: torch.size(batch_size, num_anchors, 4)
                anchors shape: torch.size(num_anchors,4)
            anchors: Priors
            targets: Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5]
                (last idx is the label, 0 for background, >0 for target).

        """
        loc_pred, conf_pred = predictions
        num = loc_pred.size(0)
        num_anchors = anchors.size(0)
        # result: match anchors (default boxes) and ground truth boxes
        # loc_t = torch.Tensor(num, num_anchors, 4)
        # conf_t = torch.LongTensor(num, num_anchors)
        # if loc_pred.is_cuda:
        #     loc_t = loc_t.cuda()
        #     conf_t = conf_t.cuda()

        loc_t = loc_pred.detach().new(num, num_anchors, 2)
        conf_t = loc_pred.detach().new(num, num_anchors).long()

        # pdb.set_trace()
        for idx in range(num):
            cur_targets = targets[idx].detach()
            valid_targets = cur_targets[cur_targets[:, -1] > 0]
            truths = valid_targets[:, :-1]
            labels = torch.ones_like(valid_targets[:, -1])
            # encode results are stored in loc_t and conf_t
            match(self.overlap_thresh, truths, anchors.detach(), self.variance,
                  labels, loc_t, conf_t, idx)

        # # wrap targets
        # loc_t = Variable(loc_t, requires_grad=False)
        # conf_t = Variable(conf_t, requires_grad=False)
        # valid indice.
        pos_loc_flag = loc_t.new(loc_t).byte()
        pos_conf_flag = conf_t.new(conf_t).byte()
        neg_conf_flag = conf_t.new(conf_t).byte()

        for idx in range(num):
            single_conf_t = conf_t[idx]
            pos = single_conf_t > 0
            pos_loc_flag[idx] = pos.unsqueeze(1).expand_as(loc_t[idx])

            # Mimic MAX_NEGATIVE of caffe-ssd
            # Compute max conf across a batch for selecting negatives with large
            # error confidence.
            single_conf_pred = conf_pred[idx]
            # Sum up losses of all wrong classes.
            # This loss is only used to select max negatives.
            loss_conf_proxy = (log_sum_exp(single_conf_pred) - single_conf_pred.gather(
                1, single_conf_t.view(-1, 1))).view(-1)
            # Exclude positives
            loss_conf_proxy[pos] = 0
            # Sort and select max negatives
            # Values in loss_c are not less than 0.
            _, loss_idx = loss_conf_proxy.sort(0, descending=True)
            _, idx_rank = loss_idx.sort(0)
            # pdb.set_trace()
            num_pos = torch.sum(pos)
            # clamp number of negtives.
            num_neg = torch.min(self.neg_pos_ratio * num_pos, pos.size(0) - num_pos)
            neg = idx_rank < num_neg.expand_as(idx_rank)
            # Total confidence loss includes positives and negatives.
            pos_conf_flag[idx] = pos
            neg_conf_flag[idx] = neg

        pos_loc_t = loc_t[pos_loc_flag.detach()].view(-1, 2)
        # # Select postives to compute bounding box loss.
        pos_loc_pred = loc_pred[pos_loc_flag.detach()].view(-1, 2)
        # loss_l = functional.smooth_l1_loss(pos_loc_pred, pos_loc_t, size_average=False)
        loss_l = F.smooth_l1_loss(pos_loc_pred, pos_loc_t, reduction='sum')
        # pdb.set_trace()
        # Final classification loss
        conf_keep = (pos_conf_flag + neg_conf_flag).view(-1).gt(0).nonzero().view(-1)
        valid_conf_pred = torch.index_select(conf_pred.view(-1, self.num_classes), 0, conf_keep)
        valid_conf_t = torch.index_select(conf_t.view(-1), 0, conf_keep)
        # loss_c = functional.cross_entropy(valid_conf_pred, valid_conf_t, size_average=False)
        loss_c = F.cross_entropy(valid_conf_pred, valid_conf_t, reduction='sum')
        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + alpha*Lloc(x,l,g)) / N
        # only number of positives
        total_num = torch.sum(pos_conf_flag).float()
        # print('arm_loss', loss_l, loss_c, total_num)
        # pdb.set_trace()
        loss_l /= total_num
        loss_c /= total_num

        return loss_l, loss_c


class RPDLoss(nn.Module):
    def __init__(self, num_classes, overlap_thresh,
                 neg_pos_ratio, variance):
        super(RPDLoss, self).__init__()
        self.num_classes = num_classes
        self.overlap_thresh = overlap_thresh
        self.neg_pos_ratio = neg_pos_ratio
        self.variance = variance

    def forward(self, odm_predictions, refined_anchors, ignore_flags_refined_anchor, targets):
        (loc_pred, conf_pred) = odm_predictions
        num = refined_anchors.size(0)
        num_anchors = refined_anchors.size(1)
        loc_t = loc_pred.detach().new(num, num_anchors, 2)
        conf_t = loc_pred.detach().new(num, num_anchors).long()

        for idx in range(num):
            cur_targets = targets[idx].detach()
            valid_targets = cur_targets[cur_targets[:, -1] > 0]
            gt_boxes = valid_targets[:, :-1]
            labels = valid_targets[:, -1]

            cur_anchors = refined_anchors[idx]
            cur_ignore_flags = ignore_flags_refined_anchor[idx].detach()
            # valid anchors.
            match_with_flags(self.overlap_thresh, gt_boxes, cur_anchors,
                             cur_ignore_flags, self.variance, labels,
                             loc_t, conf_t, idx)

        # valid indice.
        pos_loc_flag = loc_t.new(loc_t).byte()
        pos_conf_flag = conf_t.new(conf_t).byte()
        neg_conf_flag = conf_t.new(conf_t).byte()

        for idx in range(num):
            single_conf_t = conf_t[idx]
            pos = single_conf_t > 0
            pos_loc_flag[idx] = pos.unsqueeze(1).expand_as(loc_t[idx])

            single_conf_pred = conf_pred[idx]
            loss_conf_proxy = (log_sum_exp(single_conf_pred) - single_conf_pred.gather(
                1, single_conf_t.view(-1, 1))).view(-1)
            loss_conf_proxy[pos] = 0
            loss_conf_proxy[ignore_flags_refined_anchor[idx].gt(0)] = 0
            _, loss_idx = loss_conf_proxy.sort(0, descending=True)
            _, idx_rank = loss_idx.sort(0)
            num_pos = torch.sum(pos)
            max_neg = torch.sum(loss_conf_proxy > 0)
            num_neg = torch.min(self.neg_pos_ratio * num_pos, max_neg)
            neg = idx_rank < num_neg.expand_as(idx_rank)
            pos_conf_flag[idx] = pos
            neg_conf_flag[idx] = neg

        pos_loc_t = loc_t[pos_loc_flag.detach()].view(-1, 2)
        pos_loc_pred = loc_pred[pos_loc_flag.detach()].view(-1, 2)
        loss_l = F.smooth_l1_loss(pos_loc_pred, pos_loc_t, reduction='sum')
        conf_keep = (pos_conf_flag + neg_conf_flag).view(-1).gt(0).nonzero().view(-1)
        valid_conf_pred = torch.index_select(conf_pred.view(-1, self.num_classes), 0, conf_keep)
        valid_conf_t = torch.index_select(conf_t.view(-1), 0, conf_keep)
        loss_c = F.cross_entropy(valid_conf_pred, valid_conf_t, reduction='sum')
        total_num = torch.sum(pos_conf_flag).float()
        loss_l /= total_num
        loss_c /= total_num

        return loss_l, loss_c


class FGDLoss(nn.Module):
    def __init__(self, num_classes, overlap_thresh,
                 neg_pos_ratio, variance):
        super(FGDLoss, self).__init__()
        self.num_classes = num_classes
        self.overlap_thresh = overlap_thresh
        self.neg_pos_ratio = neg_pos_ratio
        self.variance = variance

    def forward(self, odm_predictions, refined_anchors, ignore_flags_refined_anchor, targets):
        (loc_pred, conf_pred) = odm_predictions
        num = refined_anchors.size(0)
        num_anchors = refined_anchors.size(1)
        loc_t = loc_pred.detach().new(num, num_anchors, 2)
        conf_t = loc_pred.detach().new(num, num_anchors).long()
        for idx in range(num):
            cur_targets = targets[idx].detach()
            valid_targets = cur_targets[cur_targets[:, -1] > 0]
            gt_boxes = valid_targets[:, :-1]
            labels = valid_targets[:, -1]

            cur_anchors = refined_anchors[idx]
            cur_ignore_flags = ignore_flags_refined_anchor[idx].detach()
            match_with_flags(self.overlap_thresh, gt_boxes, cur_anchors,
                             cur_ignore_flags, self.variance, labels,
                             loc_t, conf_t, idx)

        pos_loc_flag = loc_t.new(loc_t).byte()
        pos_conf_flag = conf_t.new(conf_t).byte()

        for idx in range(num):
            single_conf_t = conf_t[idx]
            pos = single_conf_t > 0
            pos_loc_flag[idx] = pos.unsqueeze(1).expand_as(loc_t[idx])
            single_conf_pred = conf_pred[idx]
            loss_conf_proxy = (log_sum_exp(single_conf_pred) - single_conf_pred.gather(
                1, single_conf_t.view(-1, 1))).view(-1)
            loss_conf_proxy[pos] = 0
            loss_conf_proxy[ignore_flags_refined_anchor[idx].gt(0)] = 0
            _, loss_idx = loss_conf_proxy.sort(0, descending=True)
            _, idx_rank = loss_idx.sort(0)
            pos_conf_flag[idx] = pos

        pos_loc_t = loc_t[pos_loc_flag.detach()].view(-1, 2)
        pos_loc_pred = loc_pred[pos_loc_flag.detach()].view(-1, 2)
        loss_l = F.smooth_l1_loss(pos_loc_pred, pos_loc_t, reduction='sum')
        total_num = torch.sum(pos_conf_flag).float()
        loss_l /= total_num

        return loss_l


class FLCLoss(nn.Module):
    def __init__(self, num_classes):
        super(FLCLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, logits, targets):
        targets = targets.view(-1)
        logits = logits.view(-1, self.num_classes)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt_scores, logits=pred_anchors)

        # loss = tf.reduce_mean(loss)

        pmask = tf.cast(gt_scores > 0, dtype=tf.float32)
        num_positive = tf.reduce_sum(pmask)
        num_entries = tf.cast(tf.shape(gt_scores)[0], dtype=tf.float32)
        num_positive = tf.clip_by_value(num_positive, 1., num_entries - 1.)
        ratio = num_entries / num_positive
        coef_0 = 0.5 * (ratio) / (ratio - 1)
        coef_1 = coef_0 * (ratio - 1)

        loss = (coef_1 * pmask + coef_0 * (1.0 - pmask)) * loss
        loss_c = tf.reduce_mean(loss)
        return loss_c


def binary_logistic_loss(gt_scores, pred_anchors):
    gt_scores = tf.reshape(gt_scores, [-1])
    pred_anchors = tf.reshape(pred_anchors, [-1, NUM_CLASSES])

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt_scores, logits=pred_anchors)

    # loss = tf.reduce_mean(loss)

    pmask = tf.cast(gt_scores > 0, dtype=tf.float32)
    num_positive = tf.reduce_sum(pmask)
    num_entries = tf.cast(tf.shape(gt_scores)[0], dtype=tf.float32)
    num_positive = tf.clip_by_value(num_positive, 1., num_entries - 1.)
    ratio = num_entries / num_positive
    coef_0 = 0.5 * (ratio) / (ratio - 1)
    coef_1 = coef_0 * (ratio - 1)

    loss = (coef_1 * pmask + coef_0 * (1.0 - pmask)) * loss
    loss = tf.reduce_mean(loss)
    # loss = -10 * tf.reduce_sum(loss) / num_entries
    return loss
