import torch.nn as nn
import torch.nn.functional as F
import torch
import einops


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


class BCEWithLogitsLossWithIgnoreIndex(nn.Module):
    def __init__(self, reduction='mean', ignore_index=255):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets, weight=None):
        # inputs of size B x C x H x W
        n_cl = torch.tensor(inputs.shape[1]).to(inputs.device)
        labels_new = torch.where(targets != self.ignore_index, targets, n_cl)
        # replace ignore with numclasses + 1 (to enable one hot and then remove it)
        targets = F.one_hot(labels_new, inputs.shape[1] + 1).float().permute(0, 3, 1, 2)
        targets = targets[:, :inputs.shape[1], :, :]  # remove 255 from 1hot
        # targets is B x C x H x W so shape[1] is C

        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # loss has shape B x C x H x W
        loss = loss.sum(dim=1)  # sum the contributions of the classes

        if weight is not None:
            loss = loss * weight

        if self.reduction == 'mean':
            # return loss.mean()
            # if targets have only zeros, we skip them
            return torch.masked_select(loss, targets.sum(dim=1) != 0).mean()
        elif self.reduction == 'sum':
            # return loss.sum()
            return torch.masked_select(loss, targets.sum(dim=1) != 0).sum()
        else:
            # return loss
            return loss * targets.sum(dim=1)


# loss_kd
class UnbiasedKnowledgeDistillationLoss(nn.Module):
    def __init__(self, reduction='mean', alpha=1.):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets, mask=None):

        new_cl = inputs.shape[1] - targets.shape[1]

        targets = targets * self.alpha

        new_bkg_idx = torch.tensor([0] + [x for x in range(targets.shape[1], inputs.shape[1])]).to(inputs.device)

        den = torch.logsumexp(inputs, dim=1)  # B, H, W
        outputs_no_bgk = inputs[:, 1:-new_cl] - den.unsqueeze(dim=1)  # B, OLD_CL, H, W
        outputs_bkg = torch.logsumexp(torch.index_select(inputs, index=new_bkg_idx, dim=1), dim=1) - den  # B, H, W

        labels = torch.softmax(targets, dim=1)  # B, BKG + OLD_CL, H, W

        # make the average on the classes 1/n_cl \sum{c=1..n_cl} L_c
        loss = (labels[:, 0] * outputs_bkg + (labels[:, 1:] * outputs_no_bgk).sum(dim=1)) / targets.shape[1]

        if mask is not None:
            loss = loss * mask.float()

        if self.reduction == 'mean':
            outputs = -torch.mean(loss)
        elif self.reduction == 'sum':
            outputs = -torch.sum(loss)
        else:
            outputs = -loss

        return outputs


def debug(array, str='default'):
    print(str, array.shape, array.max(), array.min())


class Conloss_proposal(nn.Module):
    """Supervised Contrastive Learning for segmentation"""

    def __init__(self, opts=None, sample_method='none', temperature=0.07, use_pseudo_label=False):
        super(Conloss_proposal, self).__init__()
        self.temperature = temperature
        self.sample_method = sample_method
        try:
            self.num = opts.merge_proposal_num
            self.opts = opts
        except:
            self.num = 20

        self.use_pseudo_label = use_pseudo_label
        if self.use_pseudo_label:
            if opts.dataset == 'voc':
                self.num = 25
            elif opts.dataset == 'ade':
                self.num = 155
            else:
                raise NotImplementedError
        # print(temperature)

    def forward(self, pre_logits, pre_logits_prev, proposal):
        """
        :pre_logits : feature with shape b C=256 h=17 w=17
        :pre_logits_prev : feature with shape b C=256 h=17 w=17
        :proposal_1hot: b h w in range(0~num) where index 'num' is dummy class
        :return loss
        """
        try:
            device = self.opts.device
        except:
            device = pre_logits.device

        proposal = (F.interpolate(
            input=proposal.float().unsqueeze(1), size=(pre_logits.shape[2], pre_logits.shape[3]),
            mode='nearest')).to(torch.long).squeeze(1)
        B, C, H, W = pre_logits.shape
        assert pre_logits.shape == pre_logits_prev.shape
        proposal = proposal.view(-1)  # bhw
        # mask without dummy proposal
        mask_undummy = (proposal >= 0) & (proposal < self.num)
        pre_logits = einops.rearrange(pre_logits, 'b c h w -> ( b h w ) c ')
        pre_logits_prev = einops.rearrange(pre_logits_prev, 'b c h w -> ( b h w ) c ')
        feature_anc = F.normalize(pre_logits[mask_undummy], dim=1)
        label_anc = proposal[mask_undummy]
        conts = F.normalize(pre_logits_prev[mask_undummy], dim=1)

        # get contrastive probablity mask with old logits
        prev_prob = torch.softmax(pre_logits_prev, dim=1)
        feature_prev_anc = prev_prob[mask_undummy]
        feature_prev_con = torch.cat([feature_prev_anc, feature_prev_anc], dim=0)

        anc_uni = torch.unique(label_anc)
        con_uni = torch.unique(label_anc)
        anc_prototype = []
        for i in anc_uni:
            mask_i = (label_anc == i)
            proto = feature_anc[mask_i].mean(dim=0)
            anc_prototype.append(proto)
        anc_prototype = torch.stack(anc_prototype, dim=0)
        con_prototype_prev = []
        for i in con_uni:
            mask_i = (label_anc == i)
            proto = conts[mask_i].mean(dim=0)
            con_prototype_prev.append(proto)
        con_prototype_prev = torch.stack(con_prototype_prev, dim=0)
        con_prototype = torch.cat([anc_prototype, con_prototype_prev], dim=0).detach()
        con_uni = torch.cat([anc_uni, con_uni], dim=0).detach()

        anchor_features = anc_prototype
        contrast_feature = con_prototype
        anchor_labels = anc_uni
        contrast_labels = con_uni

        anchor_labels = anchor_labels.view(-1, 1)  # n 1
        contrast_labels = contrast_labels.view(-1, 1)  # n_ 1

        batch_size = anchor_features.shape[0]  # b
        R = torch.eq(anchor_labels, contrast_labels.T).float().requires_grad_(False).to(device)
        positive_mask = R.clone().requires_grad_(False)
        positive_mask[:, :batch_size] -= torch.eye(batch_size).to(device)

        positive_mask = positive_mask.detach()
        negative_mask = 1 - R
        negative_mask = negative_mask.detach()

        anchor_dot_contrast = torch.div(
            torch.mm(anchor_features, contrast_feature.T),
            self.temperature)

        neg_contrast = (torch.exp(anchor_dot_contrast) * negative_mask).sum(dim=1, keepdim=True)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        anchor_dot_contrast = anchor_dot_contrast - logits_max.detach()

        pos_contrast = torch.log(torch.exp(anchor_dot_contrast)) * positive_mask - torch.log(
            torch.exp(anchor_dot_contrast) + neg_contrast) * positive_mask

        num = positive_mask.sum(dim=1)
        loss = -torch.div(pos_contrast.sum(dim=1)[num != 0], num[num != 0])
        return loss.mean()
