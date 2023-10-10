import os
from PIL import Image
import torch
import numpy as np
import torch.nn.functional as F


def proposal_preprocess(opts,proposals,need_1_hot=True):
    """
    functions to preprocess proposals(1hot & merge small segment)
    :param proposals:  b h w(with labels of 1 to 100)
    :param opts: including opts.device opts.proposal_channnel opts.merge_proposal_num
    :return: b * num * h w
    """
    # merge
    new_p = torch.zeros_like(proposals)
    try:
        num = opts.merge_proposal_num
    except:
        num = 20

    for k,proposal in enumerate(proposals):
        tmp = {}
        for i in range(opts.proposal_channel):
            cnt = torch.sum(proposal == i)
            tmp[i] = cnt

        tmp = sorted(tmp.items(),key=lambda x: x[1],reverse=True)
        for i,v in enumerate(tmp):
            if i>num:
                break
            new_p[k,proposal == v[0]] = i

    # to 1 hot
    if need_1_hot:
        proposals = new_p
        proposals = proposals.to(opts.device, dtype=torch.long, non_blocking=True)
        n_cl = torch.tensor(num).to(proposals.device)
        proposals_n = torch.where(proposals != 255, proposals, n_cl)

        proposals_1hot = F.one_hot(proposals_n, num + 1).permute(0, 3, 1, 2)
        return proposals_1hot
    # proposals_1hot = proposals_1hot[:, :-1].float()
    return new_p


def compose_label(opts,pseudo_label,proposal):

    merged = []
    all_sum_classes = sum(opts.num_classes)
    B = proposal.shape[0]
    for j in range(B):
        p = proposal[j]
        l = pseudo_label[j]


        for i in range(opts.prev_classes, all_sum_classes):
            mask = (l == i) # h w , bool

            masked_proposal = p[mask]

            num = torch.unique(masked_proposal)

            if torch.numel(num):
                for n in num:
                    p[p==n] = num[0]
        merged.append(p)

    merged =  torch.stack(merged) # b h w
    return merged
