import torch.nn as nn
from i3d import I3D

import torch
import torch.nn.functional as F
import numpy as np

from utils import *


class BasicConv3d(nn.Module):
    def __init__(self, in_planes=256, out_planes=256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1),
                 use_relu=True, use_bn=True):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        if use_relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

        # if use_bn:
        #     # self.bn = nn.BatchNorm3d(out_planes)
        #     self.bn = nn.SyncBatchNorm(out_planes)
        # else:
        #     self.bn = None

    def forward(self, x):
        x = self.conv(x)
        if self.relu:
            x = self.relu(x)
        # if self.bn:
        #     x = self.bn(x)
        return x


class BasicDeConv3d(nn.Module):
    def __init__(self, in_planes=256, out_planes=256, kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0),
                 use_relu=True, use_bn=True):
        super(BasicDeConv3d, self).__init__()
        self.conv = nn.ConvTranspose3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        if use_relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

        # if use_bn:
        #     # self.bn = nn.BatchNorm3d(out_planes)
        #     self.bn = nn.SyncBatchNorm(out_planes)
        # else:
        #     self.bn = None

    def forward(self, x):
        x = self.conv(x)
        if self.relu:
            x = self.relu(x)
        # if self.bn:
        #     x = self.bn(x)
        return x


class FusionBlock(nn.Module):
    def __init__(self, use_upsample=True):
        super(FusionBlock, self).__init__()
        self.conv0 = BasicConv3d(use_relu=False, use_bn=False)
        if use_upsample:
            self.conv1 = BasicDeConv3d(256, 256, kernel_size=(2, 1, 1), stride=(2, 1, 1), use_relu=False, use_bn=False)
            # self.conv1 = BasicConv3d(use_relu=False)
        else:
            self.conv1 = None
        self.conv2 = BasicConv3d()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        x = self.conv0(x)
        if self.conv1:
            dey = self.conv1(y)
            # dey = F.interpolate(dey, x.size()[2:])
            # dey = dey[0:x.size()[-3]]
            x = x + dey
        x = self.relu(x)
        x = self.conv2(x)
        return x


class FusionBlockV2(nn.Module):
    def __init__(self):
        super(FusionBlockV2, self).__init__()
        self.conv0 = nn.Sequential(
            BasicConv3d(3, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2)),
            BasicConv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2)),
            BasicConv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 2, 2), use_relu=False, use_bn=False),
            nn.AvgPool3d(kernel_size=(1, 4, 4), stride=(1, 4, 4), padding=(0, 0, 0)))

        self.conv1 = nn.Sequential(
            BasicDeConv3d(256, 256, kernel_size=(2, 1, 1), stride=(2, 1, 1)),
            BasicDeConv3d(256, 256, kernel_size=(2, 1, 1), stride=(2, 1, 1)),
            BasicDeConv3d(256, 256, kernel_size=(2, 1, 1), stride=(2, 1, 1), use_relu=False, use_bn=False))

        self.conv2 = BasicConv3d()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        x = self.conv0(x)
        y = self.conv1(y)
        x = x + y
        x = self.relu(x)
        x = self.conv2(x)
        return x


class PBRNet(nn.Module):
    def __init__(self, backbone, anchors, num_classes=21, in_channels=1024):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = [a.shape[-2] for a in anchors]
        self.anchors = anchors
        self.backbone = backbone

        self.cpd_layer = nn.ModuleList([
            BasicConv3d(in_planes=in_channels, stride=(1, 1, 1)),
            BasicConv3d(stride=(2, 1, 1)),
            BasicConv3d(stride=(2, 1, 1)),
            BasicConv3d(stride=(2, 1, 1)),
            BasicConv3d(stride=(2, 1, 1)),
            BasicConv3d(stride=(2, 1, 1))
        ])

        self.cpd_loc = nn.ModuleList([
            BasicConv3d(out_planes=self.num_anchors[0] * 2, padding=(1, 0, 0), use_relu=False, use_bn=False),
            BasicConv3d(out_planes=self.num_anchors[1] * 2, padding=(1, 0, 0), use_relu=False, use_bn=False),
            BasicConv3d(out_planes=self.num_anchors[2] * 2, padding=(1, 0, 0), use_relu=False, use_bn=False),
            BasicConv3d(out_planes=self.num_anchors[3] * 2, padding=(1, 0, 0), use_relu=False, use_bn=False),
            BasicConv3d(out_planes=self.num_anchors[4] * 2, padding=(1, 0, 0), use_relu=False, use_bn=False),
            BasicConv3d(out_planes=self.num_anchors[4] * 2, padding=(1, 0, 0), use_relu=False, use_bn=False)
        ])

        self.cpd_conf = nn.ModuleList([
            BasicConv3d(out_planes=self.num_anchors[0] * self.num_classes, padding=(1, 0, 0), use_relu=False,
                        use_bn=False),
            BasicConv3d(out_planes=self.num_anchors[1] * self.num_classes, padding=(1, 0, 0), use_relu=False,
                        use_bn=False),
            BasicConv3d(out_planes=self.num_anchors[2] * self.num_classes, padding=(1, 0, 0), use_relu=False,
                        use_bn=False),
            BasicConv3d(out_planes=self.num_anchors[3] * self.num_classes, padding=(1, 0, 0), use_relu=False,
                        use_bn=False),
            BasicConv3d(out_planes=self.num_anchors[4] * self.num_classes, padding=(1, 0, 0), use_relu=False,
                        use_bn=False),
            BasicConv3d(out_planes=self.num_anchors[4] * self.num_classes, padding=(1, 0, 0), use_relu=False,
                        use_bn=False)
        ])

        self.rpd_layer = nn.ModuleList([
            FusionBlock(use_upsample=False),
            FusionBlock(),
            FusionBlock(),
            FusionBlock(),
            FusionBlock(),
            FusionBlock()
        ])

        self.rpd_loc = nn.ModuleList([
            BasicConv3d(out_planes=self.num_anchors[5] * 2, padding=(1, 0, 0), use_relu=False, use_bn=False),
            BasicConv3d(out_planes=self.num_anchors[4] * 2, padding=(1, 0, 0), use_relu=False, use_bn=False),
            BasicConv3d(out_planes=self.num_anchors[3] * 2, padding=(1, 0, 0), use_relu=False, use_bn=False),
            BasicConv3d(out_planes=self.num_anchors[2] * 2, padding=(1, 0, 0), use_relu=False, use_bn=False),
            BasicConv3d(out_planes=self.num_anchors[1] * 2, padding=(1, 0, 0), use_relu=False, use_bn=False),
            BasicConv3d(out_planes=self.num_anchors[0] * 2, padding=(1, 0, 0), use_relu=False, use_bn=False)
        ])

        self.rpd_conf = nn.ModuleList([
            BasicConv3d(out_planes=self.num_anchors[5] * self.num_classes, padding=(1, 0, 0), use_relu=False,
                        use_bn=False),
            BasicConv3d(out_planes=self.num_anchors[4] * self.num_classes, padding=(1, 0, 0), use_relu=False,
                        use_bn=False),
            BasicConv3d(out_planes=self.num_anchors[3] * self.num_classes, padding=(1, 0, 0), use_relu=False,
                        use_bn=False),
            BasicConv3d(out_planes=self.num_anchors[2] * self.num_classes, padding=(1, 0, 0), use_relu=False,
                        use_bn=False),
            BasicConv3d(out_planes=self.num_anchors[1] * self.num_classes, padding=(1, 0, 0), use_relu=False,
                        use_bn=False),
            BasicConv3d(out_planes=self.num_anchors[0] * self.num_classes, padding=(1, 0, 0), use_relu=False,
                        use_bn=False)
        ])

        self.fgd_layer = FusionBlockV2()

        self.fgd_start_loc = BasicConv3d(out_planes=1, padding=(0, 0, 0), use_relu=False, use_bn=False)
        self.fgd_end_loc = BasicConv3d(out_planes=1, padding=(0, 0, 0), use_relu=False, use_bn=False)

        self.fgd_start_conf = BasicConv3d(out_planes=1, padding=(1, 0, 0), use_relu=False, use_bn=False)
        self.fgd_end_conf = BasicConv3d(out_planes=1, padding=(1, 0, 0), use_relu=False, use_bn=False)
        # self.fgd_action_conf = BasicConv3d(out_planes=num_classes, padding=(1, 0, 0), use_relu=False)

        self.softmax = nn.Softmax(-1)
        self.sigmoid = nn.Sigmoid()

        self.eps = 1e-8
        self.variances = [1, 1]

    def forward(self, inp, gt_boxes, gt_labels, gt_start, gt_end):
        B = inp.size()[0]
        x = self.backbone(inp)

        cpd_sources = []
        cpd_logits = []
        cpd_localisations = []
        for i in range(len(self.cpd_layer)):
            x = self.cpd_layer[i](x)
            cpd_sources.append(x)
            cpd_loc = self.cpd_loc[i](x).squeeze(-1).squeeze(-1)  # B, 2*N, T
            cpd_loc = cpd_loc.permute(0, 2, 1).contiguous()  # B, T, 2*N
            cpd_loc = cpd_loc.view(B, -1, 2)  # B, T*N, 2
            cpd_localisations.append(cpd_loc)

            cpd_conf = self.cpd_conf[i](x).squeeze(-1).squeeze(-1)  # B, N, T
            cpd_conf = cpd_conf.permute(0, 2, 1).contiguous()  # B, T, N
            cpd_conf = cpd_conf.view(B, -1, self.num_classes)  # B, T*N, C
            cpd_logits.append(cpd_conf)

        cpd_logits = torch.cat(cpd_logits, 1)  # B, T1*N1 + T2*N2, C
        cpd_preds = self.softmax(cpd_logits)
        cpd_localisations = torch.cat(cpd_localisations, 1)  # B, T1*N1 + T2*N2, 2
        cpd_anchors = np.concatenate([anchor.reshape([1, -1, 2]) for anchor in self.anchors], 1)  # 1, T*N, 2
        cpd_anchors = torch.from_numpy(cpd_anchors).cuda().repeat(B, 1, 1)  # B, T*N, 2
        matched_boxes, matched_labels = self.match(gt_boxes, gt_labels, cpd_anchors, threshold=0.4)
        matched_boxes = self.encode(self.center_form(matched_boxes), cpd_anchors)
        loss = {}
        cpd_pos_loss, cpd_neg_loss = self.cls_loss(
            logits=cpd_logits,
            labels=matched_labels
        )
        loss['cpd_pos'] = cpd_pos_loss
        loss['cpd_neg'] = cpd_neg_loss
        cpd_loc_loss = self.loc_loss(
            localisations=cpd_localisations,
            boxes=matched_boxes,
            mask=(matched_labels > 0))
        loss['cpd_loc'] = cpd_loc_loss
        rpd_logits = []
        rpd_localisations = []
        for i in range(len(self.rpd_layer)):
            x = self.rpd_layer[i](cpd_sources[-(i + 1)], x)

            rpd_loc = self.rpd_loc[i](x).squeeze(-1).squeeze(-1)  # B, 2*N, T
            rpd_loc = rpd_loc.permute(0, 2, 1).contiguous()  # B, T, 2*N
            rpd_loc = rpd_loc.view(B, -1, 2)  # B, T, N, 2
            rpd_localisations.append(rpd_loc)

            rpd_conf = self.rpd_conf[i](x).squeeze(-1).squeeze(-1)  # B, N, T
            rpd_conf = rpd_conf.permute(0, 2, 1).contiguous()  # B, T, N
            rpd_conf = rpd_conf.view(B, -1, self.num_classes)
            rpd_logits.append(rpd_conf)
        rpd_logits = torch.cat(rpd_logits[::-1], 1)  # B, T2*N2 + T1*N1, 2
        rpd_preds = self.softmax(rpd_logits)

        rpd_localisations = torch.cat(rpd_localisations[::-1], 1)  # B, T2*N2 + T1*N1, 2

        rpd_anchors = self.decode(cpd_localisations, cpd_anchors)
        matched_boxes, matched_labels = self.match(gt_boxes, gt_labels, rpd_anchors, threshold=0.5)
        matched_boxes = self.encode(self.center_form(matched_boxes), rpd_anchors)

        cpd_reject_mask = cpd_preds[..., 0] < 0.99

        rpd_pos_loss, rpd_neg_loss = self.cls_loss(
            logits=rpd_logits,
            labels=matched_labels,
            mask=cpd_reject_mask,
        )

        loss['rpd_pos'] = rpd_pos_loss
        loss['rpd_neg'] = rpd_neg_loss

        rpd_loc_loss = self.loc_loss(
            localisations=rpd_localisations,
            boxes=matched_boxes,
            mask=(cpd_reject_mask | (matched_labels > 0)))

        loss['rpd_loc'] = rpd_loc_loss

        x = self.fgd_layer(inp, x)  # B, C, T, 3, 3
        B, C, T, H, W = x.size()

        start_logits = self.fgd_start_conf(x)  # B, T
        end_logits = self.fgd_end_conf(x)  # B, T

        start_preds = self.sigmoid(start_logits)
        end_preds = self.sigmoid(end_logits)

        # action_logits = self.fgd_action_conf(x).view(B, C, T)   # B, C, T
        fgd_bsn_loss = self.bsn_loss(start_preds,
                                     end_preds,
                                     gt_start,
                                     gt_end)

        loss['fgd_bsn'] = fgd_bsn_loss

        fgd_anchors = self.decode(rpd_localisations, rpd_anchors)

        fgd_anchors = self.point_form(fgd_anchors)
        s = (fgd_anchors[..., 0] * T).long().clamp(0, T - 2)
        e = (fgd_anchors[..., 1] * T).long()
        e = torch.maximum(s + 1, e).clamp(1, T - 1)

        fgd_anchors = (torch.stack([s, e], -1).float() + 0.5) / T

        d = e - s
        r = d // 8  # B, N
        xp = F.pad(x, [0, 0, 0, 0, 0, 1])  # B, C, T+1, 3, 3

        ind_s = torch.stack([
            (s - r).clamp(-1, T),
            s,
            (s + r).clamp(-1, T)],
            dim=-1)  # B, N, 3

        ind_e = torch.stack([
            (e - r).clamp(-1, T),
            e,
            (e + r).clamp(-1, T)],
            dim=-1)  # B, N, 3

        ind_b = torch.arange(B).view(B, 1, 1).long().cuda()  # B, 1, 1

        x_s = xp[ind_b, :, ind_s]  # B, N, 3, C, 3, 3
        x_e = xp[ind_b, :, ind_e]  # B, N, 3, C, 3, 3

        x_s = x_s.transpose(2, 3)  # B, N, C, 3, 3, 3
        x_s = x_s.contiguous().view(-1, *x_s.size()[2:])  # B*N, C, 3, 3, 3

        x_e = x_e.transpose(2, 3)  # B, N, C, 3, 3, 3
        x_e = x_e.contiguous().view(-1, *x_e.size()[2:])  # B*N, C, 3, 3, 3

        start_locs = self.fgd_start_loc(x_s).contiguous().view(B, -1)  # B, N
        end_locs = self.fgd_end_loc(x_e).contiguous().view(B, -1)  # B, N

        bound_locs = torch.stack([start_locs, end_locs], -1)  # B, N, 2

        matched_boxes, matched_labels = self.match_s_e(gt_boxes, gt_labels, fgd_anchors, threshold=0.6)
        matched_boxes = self.encode_s_e(matched_boxes, fgd_anchors)

        rpd_reject_mask = rpd_preds[..., 0] < 0.99
        fgd_loc_loss = self.loc_loss(
            localisations=bound_locs,
            boxes=matched_boxes,
            mask=(rpd_reject_mask | (matched_labels > 0)))
        loss['fgd_loc'] = fgd_loc_loss
        fgd_locs = self.decode_s_e(bound_locs, fgd_anchors)
        return loss, cpd_logits, rpd_logits, start_logits, end_logits, fgd_locs

    def match(self, gt_boxes, gt_labels, anchors, threshold=0.5):
        overlaps = self.jaccard(self.point_form(anchors), gt_boxes)  # B, N, G
        overlaps, idx = overlaps.max(-1)  # B, N
        labels = torch.gather(gt_labels, -1, idx)  # B, N

        labels = torch.where(overlaps > threshold, labels, 0)
        boxes = torch.stack([torch.gather(gt_boxes[..., 0], -1, idx),
                             torch.gather(gt_boxes[..., 1], -1, idx)],
                            -1)  # B, N, 2
        # boxes = self.encode(self.center_form(boxes), anchors, variances)
        return boxes, labels

    def match_s_e(self, gt_boxes, gt_labels, anchors, threshold=0.5):
        overlaps = self.jaccard(self.point_form(anchors), gt_boxes)  # B, N, G
        overlaps, idx = overlaps.max(-1)  # B, N
        labels = torch.gather(gt_labels, -1, idx)  # B, N
        labels = torch.where(overlaps > threshold, labels, 0)
        boxes = torch.stack([torch.gather(gt_boxes[..., 0], -1, idx),
                             torch.gather(gt_boxes[..., 1], -1, idx)],
                            -1)  # B, N, 2
        # boxes = self.encode_s_e(boxes, self.point_form(anchors), variances)
        return boxes, labels

    def jaccard(self, boxes0, boxes1):
        boxes0 = boxes0.unsqueeze(-2)  # B, N0, 1, 2
        boxes1 = boxes1.unsqueeze(-3)  # B, 1, N1, 2
        inner = (torch.min(boxes0[..., 1], boxes1[..., 1]) - torch.max(boxes0[..., 0], boxes1[..., 0]))
        union = (torch.max(boxes0[..., 1], boxes1[..., 1]) - torch.min(boxes0[..., 0], boxes1[..., 0]))

        # overlaps = (d0 > self.eps).float() * (d1 > self.eps).float() * overlaps  # B, N0, N1
        overlaps = inner.clamp(0., ) / union.clamp(self.eps, )
        return overlaps

    def point_form(self, boxes):
        return torch.cat((boxes[..., :1] - boxes[..., 1:] / 2,
                          boxes[..., :1] + boxes[..., 1:] / 2), -1)

    def center_form(self, boxes):
        return torch.cat(((boxes[..., 1:] + boxes[..., :1]) / 2,
                          boxes[..., 1:] - boxes[..., :1]), -1)

    def encode(self, matched, anchors):
        g_cxcy = matched[..., :1] - anchors[..., :1]
        g_cxcy /= self.variances[0] * anchors[..., 1:].clamp(self.eps)
        g_wh = matched[..., 1:] / anchors[..., 1:].clamp(self.eps)
        g_wh = torch.log(g_wh + self.eps) / self.variances[1]
        return torch.cat([g_cxcy, g_wh], -1)  # [num_anchors,2]

    def decode(self, loc, anchors):
        boxes = torch.cat((anchors[..., :1] + loc[..., :1] * self.variances[0] * anchors[..., 1:],
                           anchors[..., 1:] * torch.exp(loc[..., 1:] * self.variances[1])), -1)
        return boxes

    def encode_s_e(self, matched, anchors):
        return (matched - anchors) / (self.variances[0] * (anchors[..., 1:] - anchors[..., :1]))

    def decode_s_e(self, loc, anchors):
        return anchors + loc * self.variances[0] * (anchors[..., 1:] - anchors[..., :1])

    def cls_loss(self, logits, labels, ratio=1., mask=None):
        logits = logits.contiguous().view(-1, logits.size()[-1])
        labels = labels.contiguous().view(-1, )
        if mask is not None:
            mask = mask.contiguous().view(-1, )
            logits = logits[mask]
            labels = labels[mask]
        loss = F.cross_entropy(logits, labels, reduction='none')
        pmask = labels > 0
        nmask = torch.logical_not(pmask)
        pos_loss = loss[pmask]
        neg_loss = loss[nmask]
        num_pos = pmask.sum().item()
        num_neg = nmask.sum().item()
        num_neg = min(ratio * num_pos, num_neg)
        neg_loss, _ = neg_loss.topk(int(num_neg))
        return pos_loss.mean(), neg_loss.mean()

    def loc_loss(self, localisations, boxes, mask=None):
        localisations = localisations.contiguous().view(-1, 2)
        boxes = boxes.contiguous().view(-1, 2)
        if mask is not None:
            mask = mask.contiguous().view(-1, )
            localisations = localisations[mask]
            boxes = boxes[mask]
        smooth_loss = F.smooth_l1_loss(boxes, localisations, reduction='none')
        return smooth_loss.mean()

    def bsn_loss(self, pred_start, pred_end, gt_start, gt_end):
        def bi_loss(pred_score, gt_label):
            pred_score = pred_score.contiguous().view(-1)
            gt_label = gt_label.contiguous().view(-1)
            pmask = (gt_label > 0.5).float()
            num_entries = pred_score.size(0)
            num_positive = torch.sum(pmask)
            ratio = num_entries / num_positive
            coef_0 = 0.5 * ratio / (ratio - 1)
            coef_1 = 0.5 * ratio
            loss_pos = torch.log(pred_score + self.eps) * pmask
            loss_pos = coef_1 * torch.mean(loss_pos)
            loss_neg = torch.log(1.0 - pred_score + self.eps) * (1.0 - pmask)
            loss_neg = coef_0 * torch.mean(loss_neg)
            loss = -1 * (loss_pos + loss_neg)
            return loss

        loss_start = bi_loss(pred_start, gt_start)
        loss_end = bi_loss(pred_end, gt_end)
        loss = loss_start + loss_end
        return loss
