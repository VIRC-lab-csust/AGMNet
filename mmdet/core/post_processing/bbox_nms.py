import torch
from mmcv.ops.nms import batched_nms

from mmdet.core.bbox.iou_calculators import bbox_overlaps


def multiclass_nms(multi_bboxes,
                   multi_scores_class,
                   multi_scores_rust,
                   multi_scores_state,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels \
            are 0-based.
    """
    num_classes = multi_scores_class.size(1) - 1
    num_rusts = multi_scores_rust.size(1) - 1
    num_states = multi_scores_state.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores_class.size(0), -1, 4)#(1000,2,4)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores_class.size(0), num_classes, 4)

    scores_class = multi_scores_class[:, :-1]#(1000,2)
    scores_rust = multi_scores_rust[:, :-1]#(1000,4)
    scores_state = multi_scores_state[:, :-1]#(1000,3)

    if score_factors is not None:
        scores = scores * score_factors[:, None]

    labels_class = torch.arange(num_classes, dtype=torch.long)
    labels_rust = torch.arange(num_rusts, dtype=torch.long)
    labels_state = torch.arange(num_states, dtype=torch.long)
    labels_class = labels_class.view(1, -1).expand_as(scores_class)#(1000,2)
    labels_rust = labels_rust.view(1, -1).expand_as(scores_rust)#(1000,4)
    labels_state = labels_state.view(1, -1).expand_as(scores_state)#(1000,3)

    bboxes = bboxes.reshape(-1, 4)#(2000,4)
    scores_class = scores_class.reshape(-1)#2000
    labels_class = labels_class.reshape(-1)#2000
    scores_rust = scores_rust.reshape(-1)
    labels_rust = labels_rust.reshape(-1)
    scores_state = scores_state.reshape(-1)
    labels_state = labels_state.reshape(-1)

    # remove low scoring boxes
    valid_mask_c = scores_class > score_thr
    valid_mask_r = scores_rust > score_thr
    valid_mask_s = scores_state > score_thr
    #valid_mask = valid_mask_c * valid_mask_r * valid_mask_s

    inds_c = valid_mask_c.nonzero(as_tuple=False).squeeze(1)
    inds_r = valid_mask_r.nonzero(as_tuple=False).squeeze(1)
    inds_s = valid_mask_s.nonzero(as_tuple=False).squeeze(1)
    #inds = valid_mask.nonzero(as_tuple=False).squeeze(1)

    bboxes_class, scores_class, labels_class = bboxes[inds_c], scores_class[inds_c], labels_class[inds_c]
    bboxes_rust, scores_rust, labels_rust = bboxes[inds_r], scores_rust[inds_r], labels_rust[inds_r]
    bboxes_state, scores_state, labels_state = bboxes[inds_s], scores_state[inds_s], labels_state[inds_s]
    if inds_c.numel() == 0 or inds_s.numel() == 0 or inds_r.numel() == 0 :
        if torch.onnx.is_in_onnx_export():
            raise RuntimeError('[ONNX Error] Can not record NMS '
                               'as it has not been executed this time')
        return bboxes_class, bboxes_rust, bboxes_state, labels_class, labels_rust, labels_state

    # TODO: add size check before feed into batched_nms
    dets_c, keep_c = batched_nms(bboxes_class, scores_class, labels_class, nms_cfg)
    dets_r, keep_r = batched_nms(bboxes_rust, scores_rust, labels_rust, nms_cfg)
    dets_s, keep_s = batched_nms(bboxes_state, scores_state, labels_state, nms_cfg)
    #print(labels_class)
    #print(labels_rust)
    #print(labels_state)

    if max_num > 0:
        dets_c = dets_c[:max_num]
        dets_r = dets_r[:max_num]
        dets_s = dets_s[:max_num]
        keep_c = keep_c[:max_num]
        keep_r = keep_r[:max_num]
        keep_s = keep_s[:max_num]

    return dets_c, dets_r, dets_s, labels_class[keep_c], labels_rust[keep_r], labels_state[keep_s]


def fast_nms(multi_bboxes,
             multi_scores,
             multi_coeffs,
             score_thr,
             iou_thr,
             top_k,
             max_num=-1):
    """Fast NMS in `YOLACT <https://arxiv.org/abs/1904.02689>`_.

    Fast NMS allows already-removed detections to suppress other detections so
    that every instance can be decided to be kept or discarded in parallel,
    which is not possible in traditional NMS. This relaxation allows us to
    implement Fast NMS entirely in standard GPU-accelerated matrix operations.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class+1), where the last column
            contains scores of the background class, but this will be ignored.
        multi_coeffs (Tensor): shape (n, #class*coeffs_dim).
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        iou_thr (float): IoU threshold to be considered as conflicted.
        top_k (int): if there are more than top_k bboxes before NMS,
            only top top_k will be kept.
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept. If -1, keep all the bboxes.
            Default: -1.

    Returns:
        tuple: (bboxes, labels, coefficients), tensors of shape (k, 5), (k, 1),
            and (k, coeffs_dim). Labels are 0-based.
    """

    scores = multi_scores[:, :-1].t()  # [#class, n]
    scores, idx = scores.sort(1, descending=True)

    idx = idx[:, :top_k].contiguous()
    scores = scores[:, :top_k]  # [#class, topk]
    num_classes, num_dets = idx.size()
    boxes = multi_bboxes[idx.view(-1), :].view(num_classes, num_dets, 4)
    coeffs = multi_coeffs[idx.view(-1), :].view(num_classes, num_dets, -1)

    iou = bbox_overlaps(boxes, boxes)  # [#class, topk, topk]
    iou.triu_(diagonal=1)
    iou_max, _ = iou.max(dim=1)

    # Now just filter out the ones higher than the threshold
    keep = iou_max <= iou_thr

    # Second thresholding introduces 0.2 mAP gain at negligible time cost
    keep *= scores > score_thr

    # Assign each kept detection to its corresponding class
    classes = torch.arange(
        num_classes, device=boxes.device)[:, None].expand_as(keep)
    classes = classes[keep]

    boxes = boxes[keep]
    coeffs = coeffs[keep]
    scores = scores[keep]

    # Only keep the top max_num highest scores across all classes
    scores, idx = scores.sort(0, descending=True)
    if max_num > 0:
        idx = idx[:max_num]
        scores = scores[:max_num]

    classes = classes[idx]
    boxes = boxes[idx]
    coeffs = coeffs[idx]

    cls_dets = torch.cat([boxes, scores[:, None]], dim=1)
    return cls_dets, classes, coeffs
