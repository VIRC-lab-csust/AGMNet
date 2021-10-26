__author__ = 'tsungyi'

import copy
import datetime
import time
import torch
from collections import defaultdict

import numpy as np

from . import mask as maskUtils


class COCOeval:
    # Interface for evaluating detection on the Microsoft COCO dataset.
    #
    # The usage for CocoEval is as follows:
    #  cocoGt=..., cocoDt=...       # load dataset and results
    #  E = CocoEval(cocoGt,cocoDt); # initialize CocoEval object
    #  E.params.recThrs = ...;      # set parameters as desired
    #  E.evaluate();                # run per image evaluation
    #  E.accumulate();              # accumulate per image results
    #  E.summarize();               # display summary metrics of results
    # For example usage see evalDemo.m and http://mscoco.org/.
    #
    # The evaluation parameters are as follows (defaults in brackets):
    #  imgIds     - [all] N img ids to use for evaluation
    #  catIds     - [all] K cat ids to use for evaluation
    #  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
    #  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
    #  areaRng    - [...] A=4 object area ranges for evaluation
    #  maxDets    - [1 10 100] M=3 thresholds on max detections per image
    #  iouType    - ['segm'] set iouType to 'segm', 'bbox' or 'keypoints'
    #  iouType replaced the now DEPRECATED useSegm parameter.
    #  useCats    - [1] if true use category labels for evaluation
    # Note: if useCats=0 category labels are ignored as in proposal scoring.
    # Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.
    #
    # evaluate(): evaluates detections on every image and every category and
    # concats the results into the "evalImgs" with fields:
    #  dtIds      - [1xD] id for each of the D detections (dt)
    #  gtIds      - [1xG] id for each of the G ground truths (gt)
    #  dtMatches  - [TxD] matching gt id at each IoU or 0
    #  gtMatches  - [TxG] matching dt id at each IoU or 0
    #  dtScores   - [1xD] confidence of each dt
    #  gtIgnore   - [1xG] ignore flag for each gt
    #  dtIgnore   - [TxD] ignore flag for each dt at each IoU
    #
    # accumulate(): accumulates the per-image, per-category evaluation
    # results in "evalImgs" into the dictionary "eval" with fields:
    #  params     - parameters used for evaluation
    #  date       - date evaluation was performed
    #  counts     - [T,R,K,A,M] parameter dimensions (see above)
    #  precision  - [TxRxKxAxM] precision for every evaluation setting
    #  recall     - [TxKxAxM] max recall for every evaluation setting
    # Note: precision and recall==-1 for settings with no gt objects.
    #
    # See also coco, mask, pycocoDemo, pycocoEvalDemo
    #
    # Microsoft COCO Toolbox.      version 2.0
    # Data, paper, and tutorials available at:  http://mscoco.org/
    # Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
    # Licensed under the Simplified BSD License [see coco/license.txt]
    def __init__(self, cocoGt=None, cocoDt_c=None, cocoDt_r=None, cocoDt_s=None, iouType='segm'):
        '''
        Initialize CocoEval using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None
        '''
        if not iouType:
            print('iouType not specified. use default iouType segm')
        self.cocoGt = cocoGt  # ground truth COCO API
        self.cocoDt_c = cocoDt_c  # detections COCO API
        self.cocoDt_r = cocoDt_r  # detections COCO API
        self.cocoDt_s = cocoDt_s  # detections COCO API
        self.evalclassImgs = defaultdict(
            list)  # per-image per-category evaluation results [KxAxI] elements
        self.evalrustImgs = defaultdict(
            list)  # per-image per-category evaluation results [KxAxI] elements
        self.evalstate0Imgs = defaultdict(
            list)  # per-image per-category evaluation results [KxAxI] elements
        self.eval = {}  # accumulated evaluation results
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        self.params = Params(iouType=iouType)  # parameters
        self._paramsEval = {}  # parameters for evaluation
        self.stats = []  # result summarization
        self.ious_class = {}  # ious between all gts and dts
        self.ious_rust = {}
        self.ious_state = {}
        if cocoGt is not None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catclassIds = sorted(cocoGt.getCatIds(categories='categories_class'))
            self.params.catrustIds = sorted(cocoGt.getCatIds(categories='categories_rust'))
            self.params.catstateIds = sorted(cocoGt.getCatIds(categories='categories_state'))

    def _prepare(self):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        '''
        def _toMask(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = coco.annToRLE(ann)
                ann['segmentation'] = rle

        p = self.params
        if p.useCats:
            gts_class = self.cocoGt.loadAnns(
                self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catclassIds, name='class'))
            gts_rust = self.cocoGt.loadAnns(
                self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catrustIds, name='rust'))
            gts_state = self.cocoGt.loadAnns(
                self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catstateIds, name='state'))
            dts_class = self.cocoDt_c.loadAnns(
                self.cocoDt_c.getAnnIds(imgIds=p.imgIds, catIds=p.catclassIds, name='class'))
            dts_rust = self.cocoDt_r.loadAnns(
                self.cocoDt_r.getAnnIds(imgIds=p.imgIds, catIds=p.catrustIds, name='rust'))
            dts_state = self.cocoDt_s.loadAnns(
                self.cocoDt_s.getAnnIds(imgIds=p.imgIds, catIds=p.catstateIds, name='state'))
        else:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # convert ground truth to mask if iouType == 'segm'
        if p.iouType == 'segm':
            _toMask(gts, self.cocoGt)
            _toMask(dts, self.cocoDt)

        # set ignore flag
        for gt_cls in gts_class:
            gt_cls['ignore'] = gt_cls['ignore'] if 'ignore' in gt_cls else 0
            gt_cls['ignore'] = 'iscrowd' in gt_cls and gt_cls['iscrowd']
            if p.iouType == 'keypoints':
                gt_cls['ignore'] = (gt_cls['num_keypoints'] == 0) or gt_cls['ignore']
        self._gts_class = defaultdict(list)  # gt for evaluation
        self._dts_class = defaultdict(list)  # dt for evaluation
        for gt_cls in gts_class:
            self._gts_class[gt_cls['image_id'], gt_cls['category_class_id']].append(gt_cls)
        for dt_cls in dts_class:
            self._dts_class[dt_cls['image_id'], dt_cls['category_class_id']].append(dt_cls)

        for gt_rust in gts_rust:
            gt_rust['ignore'] = gt_rust['ignore'] if 'ignore' in gt_rust else 0
            gt_rust['ignore'] = 'iscrowd' in gt_rust and gt_rust['iscrowd']
            if p.iouType == 'keypoints':
                gt_rust['ignore'] = (gt_rust['num_keypoints'] == 0) or gt_rust['ignore']
        self._gts_rust = defaultdict(list)  # gt for evaluation
        self._dts_rust = defaultdict(list)  # dt for evaluation
        for gt_rust in gts_rust:
            self._gts_rust[gt_rust['image_id'], gt_rust['category_rust_id']].append(gt_rust)
        for dt_rust in dts_rust:
            self._dts_rust[dt_rust['image_id'], dt_rust['category_rust_id']].append(dt_rust)

        for gt_state in gts_state:
            gt_state['ignore'] = gt_state['ignore'] if 'ignore' in gt_state else 0
            gt_state['ignore'] = 'iscrowd' in gt_state and gt_state['iscrowd']
            if p.iouType == 'keypoints':
                gt_state['ignore'] = (gt_state['num_keypoints'] == 0) or gt_state['ignore']
        self._gts_state = defaultdict(list)  # gt for evaluation
        self._dts_state = defaultdict(list)  # dt for evaluation
        for gt_state in gts_state:
            self._gts_state[gt_state['image_id'], gt_state['category_state_id']].append(gt_state)
        for dt_state in dts_state:
            self._dts_state[dt_state['image_id'], dt_state['category_state_id']].append(dt_state)

        self.evalclassImgs = defaultdict(
            list)  # per-image per-category evaluation results
        self.evalrustImgs = defaultdict(
            list)
        self.evalstateImgs = defaultdict(
            list)
        self.eval = {}  # accumulated evaluation results

    def evaluate(self):
        '''
        Run per image evaluation on given images and store results
         (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        print('Running per image evaluation...')
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if p.useSegm is not None:
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
            print('useSegm (deprecated) is not None. Running {} evaluation'.
                  format(p.iouType))
        print('Evaluate annotation type *{}*'.format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catclassIds = list(np.unique(p.catclassIds))
        if p.useCats:
            p.catrustIds = list(np.unique(p.catrustIds))
        if p.useCats:
            p.catstateIds = list(np.unique(p.catstateIds))
        p.maxDets = sorted(p.maxDets)
        self.params = p

        self._prepare()
        # loop through images, area range, max detection number
        catclassIds = p.catclassIds if p.useCats else [-1]
        catrustIds = p.catrustIds if p.useCats else [-1]
        catstateIds = p.catstateIds if p.useCats else [-1]

        if p.iouType == 'segm' or p.iouType == 'bbox':
            computeclassIoU = self.computeclassIoU
            computerustIoU = self.computerustIoU
            computestateIoU = self.computestateIoU
        elif p.iouType == 'keypoints':
            computeIoU = self.computeOks

        self.ious_class = {(imgId, catId): computeclassIoU(imgId, catId)
                     for imgId in p.imgIds for catId in catclassIds}
        self.ious_rust = {(imgId, catId): computerustIoU(imgId, catId)
                     for imgId in p.imgIds for catId in catrustIds}
        self.ious_state = {(imgId, catId): computestateIoU(imgId, catId)
                     for imgId in p.imgIds for catId in catstateIds}

        #print('self.ious_class:', self.ious_class)
        #print('self.ious_rust:', self.ious_rust)
        #print('self.ious_state:', self.ious_state)

        evaluateclassImg = self.evaluateclassImg
        evaluaterustImg = self.evaluaterustImg
        evaluatestateImg = self.evaluatestateImg
        maxDet = p.maxDets[-1]

        self.evalclassImgs = [
            evaluateclassImg(imgId, catId, areaRng, maxDet) 
                        for catId in catclassIds for areaRng in p.areaRng 
                        for imgId in p.imgIds 
        ]
        self.evalrustImgs = [
            evaluaterustImg(imgId, catId, areaRng, maxDet) 
                        for catId in catrustIds for areaRng in p.areaRng 
                        for imgId in p.imgIds 
        ]
        self.evalstateImgs = [
            evaluatestateImg(imgId, catId, areaRng, maxDet) 
                        for catId in catstateIds for areaRng in p.areaRng 
                        for imgId in p.imgIds 
        ]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc - tic))

    def computeclassIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt_class = self._gts_class[imgId, catId]
            dt_class = self._dts_class[imgId, catId]
        else:
            gt = [_ for cId in p.catclassIds for _ in self._gts_class[imgId, cId]]
            dt = [_ for cId in p.catclassIds for _ in self._dts_class[imgId, cId]]
        if len(gt_class) == 0 and len(dt_class) == 0:
            return []
        inds = np.argsort([-d['score'] for d in dt_class], kind='mergesort')
        dt_class = [dt_class[i] for i in inds]
        if len(dt_class) > p.maxDets[-1]:
            dt_class = dt_class[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt_class]
            d = [d['segmentation'] for d in dt_class]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt_class]
            d = [d['bbox'] for d in dt_class]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt_class]
        ious = maskUtils.iou(d, g, iscrowd)
        return ious

    def computerustIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt_rust = self._gts_rust[imgId, catId]
            dt_rust = self._dts_rust[imgId, catId]
        else:
            gt = [_ for cId in p.catrustIds for _ in self._gts_rust[imgId, cId]]
            dt = [_ for cId in p.catrustIds for _ in self._dts_rust[imgId, cId]]
        #print('gt_rust:',gt_rust)
        #print('dt_rust:',dt_rust)
        if len(gt_rust) == 0 and len(dt_rust) == 0:
            return []
        inds = np.argsort([-d['score'] for d in dt_rust], kind='mergesort')
        dt_rust = [dt_rust[i] for i in inds]
        if len(dt_rust) > p.maxDets[-1]:
            dt_rust = dt_rust[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt_rust]
            d = [d['segmentation'] for d in dt_rust]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt_rust]
            d = [d['bbox'] for d in dt_rust]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt_rust]
        ious = maskUtils.iou(d, g, iscrowd)
        return ious

    def computestateIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt_state = self._gts_state[imgId, catId]
            dt_state = self._dts_state[imgId, catId]
        else:
            gt = [_ for cId in p.catstateIds for _ in self._gts_state[imgId, cId]]
            dt = [_ for cId in p.catstateIds for _ in self._dts_state[imgId, cId]]
        if len(gt_state) == 0 and len(dt_state) == 0:
            return []
        inds = np.argsort([-d['score'] for d in dt_state], kind='mergesort')
        dt_state = [dt_state[i] for i in inds]
        if len(dt_state) > p.maxDets[-1]:
            dt_state = dt_state[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt_state]
            d = [d['segmentation'] for d in dt_state]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt_state]
            d = [d['bbox'] for d in dt_state]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt_state]
        ious = maskUtils.iou(d, g, iscrowd)
        return ious

    def computeOks(self, imgId, catId):
        p = self.params
        # dimention here should be Nxm
        gts = self._gts[imgId, catId]
        dts = self._dts[imgId, catId]
        inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
        dts = [dts[i] for i in inds]
        if len(dts) > p.maxDets[-1]:
            dts = dts[0:p.maxDets[-1]]
        # if len(gts) == 0 and len(dts) == 0:
        if len(gts) == 0 or len(dts) == 0:
            return []
        ious = np.zeros((len(dts), len(gts)))
        sigmas = p.kpt_oks_sigmas
        vars = (sigmas * 2)**2
        k = len(sigmas)
        # compute oks between each detection and ground truth object
        for j, gt in enumerate(gts):
            # create bounds for ignore regions(double the gt bbox)
            g = np.array(gt['keypoints'])
            xg = g[0::3]
            yg = g[1::3]
            vg = g[2::3]
            k1 = np.count_nonzero(vg > 0)
            bb = gt['bbox']
            x0 = bb[0] - bb[2]
            x1 = bb[0] + bb[2] * 2
            y0 = bb[1] - bb[3]
            y1 = bb[1] + bb[3] * 2
            for i, dt in enumerate(dts):
                d = np.array(dt['keypoints'])
                xd = d[0::3]
                yd = d[1::3]
                if k1 > 0:
                    # measure the per-keypoint distance if keypoints visible
                    dx = xd - xg
                    dy = yd - yg
                else:
                    # measure minimum distance to keypoints in (x0,y0) &
                    # (x1,y1)
                    z = np.zeros((k))
                    dx = np.max((z, x0 - xd), axis=0) + np.max(
                        (z, xd - x1), axis=0)
                    dy = np.max((z, y0 - yd), axis=0) + np.max(
                        (z, yd - y1), axis=0)
                e = (dx**2 + dy**2) / vars / (gt['area'] + np.spacing(1)) / 2
                if k1 > 0:
                    e = e[vg > 0]
                ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
        return ious

    def evaluateclassImg(self, imgId, catId, aRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        p = self.params
        if p.useCats:
            gt = self._gts_class[imgId, catId]
            dt = self._dts_class[imgId, catId]
        else:
            gt = [_ for cId in p.catclassIds for _ in self._gts_class[imgId, cId]]
            dt = [_ for cId in p.catclassIds for _ in self._dts_class[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return None

        for g in gt:
            if g['ignore'] or (g['area'] < aRng[0] or g['area'] > aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        #print(self.ious_class[imgId, catId])
        #ious = self.ious_class[imgId, catId][:, gtind]
        ious = self.ious_class[imgId, catId][:, gtind] if len(
            self.ious_class[imgId, catId]) > 0 else self.ious_class[imgId, catId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm = np.zeros((T, G))
        dtm = np.zeros((T, D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T, D))
        if not len(ious) == 0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t, 1 - 1e-10])
                    m = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind, gind] > 0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind, gind] < iou:
                            continue
                        # if match successful and best so far, store
                        # appropriately
                        iou = ious[dind, gind]
                        m = gind
                    # if match made store id of match for both dt and gt
                    if m == -1:
                        continue
                    dtIg[tind, dind] = gtIg[m]
                    dtm[tind, dind] = gt[m]['id']
                    gtm[tind, m] = d['id']
        # set unmatched detections outside of area range to ignore
        a = np.array([d['area'] < aRng[0] or d['area'] > aRng[1]
                      for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm == 0, np.repeat(a, T,
                                                                      0)))
        # store results for given image and category
        return {
            'image_id': imgId,
            'category_class_id': catId,
            'aRng': aRng,
            'maxDet': maxDet,
            'dtIds': [d['id'] for d in dt],
            'gtIds': [g['id'] for g in gt],
            'dtMatches': dtm,
            'gtMatches': gtm,
            'dtScores': [d['score'] for d in dt],
            'gtIgnore': gtIg,
            'dtIgnore': dtIg,
        }

    def evaluaterustImg(self, imgId, catId, aRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        p = self.params
        if p.useCats:
            gt = self._gts_rust[imgId, catId]
            dt = self._dts_rust[imgId, catId]
        else:
            gt = [_ for cId in p.catrustIds for _ in self._gts_rust[imgId, cId]]
            dt = [_ for cId in p.catrustIds for _ in self._dts_rust[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return None

        for g in gt:
            if g['ignore'] or (g['area'] < aRng[0] or g['area'] > aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        #ious = self.ious_rust[imgId, catId][:, gtind]
        ious = self.ious_rust[imgId, catId][:, gtind] if len(
            self.ious_rust[imgId, catId]) > 0 else self.ious_rust[imgId, catId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm = np.zeros((T, G))
        dtm = np.zeros((T, D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T, D))
        if not len(ious) == 0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t, 1 - 1e-10])
                    m = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind, gind] > 0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind, gind] < iou:
                            continue
                        # if match successful and best so far, store
                        # appropriately
                        iou = ious[dind, gind]
                        m = gind
                    # if match made store id of match for both dt and gt
                    if m == -1:
                        continue
                    dtIg[tind, dind] = gtIg[m]
                    dtm[tind, dind] = gt[m]['id']
                    gtm[tind, m] = d['id']
        # set unmatched detections outside of area range to ignore
        a = np.array([d['area'] < aRng[0] or d['area'] > aRng[1]
                      for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm == 0, np.repeat(a, T,
                                                                      0)))
        # store results for given image and category
        return {
            'image_id': imgId,
            'category_rust_id': catId,
            'aRng': aRng,
            'maxDet': maxDet,
            'dtIds': [d['id'] for d in dt],
            'gtIds': [g['id'] for g in gt],
            'dtMatches': dtm,
            'gtMatches': gtm,
            'dtScores': [d['score'] for d in dt],
            'gtIgnore': gtIg,
            'dtIgnore': dtIg,
        }

    def evaluatestateImg(self, imgId, catId, aRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        p = self.params
        if p.useCats:
            gt = self._gts_state[imgId, catId]
            dt = self._dts_state[imgId, catId]
        else:
            gt = [_ for cId in p.catstateIds for _ in self._gts_state[imgId, cId]]
            dt = [_ for cId in p.catstateIds for _ in self._dts_state[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return None

        for g in gt:
            if g['ignore'] or (g['area'] < aRng[0] or g['area'] > aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        ious = self.ious_state[imgId, catId][:, gtind] if len(
            self.ious_state[imgId, catId]) > 0 else self.ious_state[imgId, catId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm = np.zeros((T, G))
        dtm = np.zeros((T, D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T, D))
        if not len(ious) == 0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t, 1 - 1e-10])
                    m = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind, gind] > 0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind, gind] < iou:
                            continue
                        # if match successful and best so far, store
                        # appropriately
                        iou = ious[dind, gind]
                        m = gind
                    # if match made store id of match for both dt and gt
                    if m == -1:
                        continue
                    dtIg[tind, dind] = gtIg[m]
                    dtm[tind, dind] = gt[m]['id']
                    gtm[tind, m] = d['id']
        # set unmatched detections outside of area range to ignore
        a = np.array([d['area'] < aRng[0] or d['area'] > aRng[1]
                      for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm == 0, np.repeat(a, T,
                                                                      0)))
        # store results for given image and category
        return {
            'image_id': imgId,
            'category_state_id': catId,
            'aRng': aRng,
            'maxDet': maxDet,
            'dtIds': [d['id'] for d in dt],
            'gtIds': [g['id'] for g in gt],
            'dtMatches': dtm,
            'gtMatches': gtm,
            'dtScores': [d['score'] for d in dt],
            'gtIgnore': gtIg,
            'dtIgnore': dtIg,
        }


    def accumulate(self, p=None):
        '''
        Accumulate per image evaluation results and store the result in
        self.eval

        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalclassImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catclassIds = p.catclassIds if p.useCats == 1 else [-1]
        p.catrustIds = p.catrustIds if p.useCats == 1 else [-1]
        p.catstateIds = p.catstateIds if p.useCats == 1 else [-1]
        T = len(p.iouThrs)
        R = len(p.recThrs)
        Kcls = len(p.catclassIds) if p.useCats else 1
        Krst = len(p.catrustIds) if p.useCats else 1
        Ksta = len(p.catstateIds) if p.useCats else 1
        A = len(p.areaRng)
        M = len(p.maxDets)
        precision_cls = -np.ones(
            (T, R, Kcls, A, M))  # -1 for the precision of absent categories
        recall_cls = -np.ones((T, Kcls, A, M))
        scores_cls = -np.ones((T, R, Kcls, A, M))

        precision_rst = -np.ones(
            (T, R, Krst, A, M))  # -1 for the precision of absent categories
        recall_rst = -np.ones((T, Krst, A, M))
        scores_rst = -np.ones((T, R, Krst, A, M))

        precision_sta = -np.ones(
            (T, R, Ksta, A, M))  # -1 for the precision of absent categories
        recall_sta = -np.ones((T, Ksta, A, M))
        scores_sta = -np.ones((T, R, Ksta, A, M))

        # create dictionary for future indexing
        _pe = self._paramsEval
        catclassIds = _pe.catclassIds if _pe.useCats else [-1]
        catrustIds = _pe.catrustIds if _pe.useCats else [-1]
        catstateIds = _pe.catstateIds if _pe.useCats else [-1]
        setKcls = set(catclassIds)
        setKrst = set(catrustIds)
        setKsta = set(catstateIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        kcls_list = [n for n, k in enumerate(p.catclassIds) if k in setKcls]
        krst_list = [n for n, k in enumerate(p.catrustIds) if k in setKrst]
        ksta_list = [n for n, k in enumerate(p.catstateIds) if k in setKsta]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [
            n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng))
            if a in setA
        ]
        i_list = [n for n, i in enumerate(p.imgIds) if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        #print('evalclassImgs:', self.evalclassImgs)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(kcls_list):
            Nk = k0 * A0 * I0
            for a, a0 in enumerate(a_list):
                Na = a0 * I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalclassImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if e is not None]
                    #print('E:', E)
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate(
                        [e['dtScores'][0:maxDet] for e in E])

                    # different sorting method generates slightly different
                    # results. mergesort is used to be consistent as Matlab
                    # implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]

                    dtm = np.concatenate(
                        [e['dtMatches'][:, 0:maxDet] for e in E], axis=1)[:,
                                                                          inds]
                    dtIg = np.concatenate(
                        [e['dtIgnore'][:, 0:maxDet] for e in E], axis=1)[:,
                                                                         inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg == 0)
                    if npig == 0:
                        continue
                    tps = np.logical_and(dtm, np.logical_not(dtIg))
                    fps = np.logical_and(np.logical_not(dtm),
                                         np.logical_not(dtIg))

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp + tp + np.spacing(1))
                        q = np.zeros((R, ))
                        ss = np.zeros((R, ))

                        if nd:
                            recall_cls[t, k, a, m] = rc[-1]
                        else:
                            recall_cls[t, k, a, m] = 0

                        # numpy is slow without cython optimization for
                        # accessing elements use python array gets significant
                        # speed improvement
                        pr = pr.tolist()
                        q = q.tolist()

                        for i in range(nd - 1, 0, -1):
                            if pr[i] > pr[i - 1]:
                                pr[i - 1] = pr[i]

                        inds = np.searchsorted(rc, p.recThrs, side='left')
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except:  # noqa: E722
                            pass
                        precision_cls[t, :, k, a, m] = np.array(q)
                        scores_cls[t, :, k, a, m] = np.array(ss)

        for k, k0 in enumerate(krst_list):
            Nk = k0 * A0 * I0
            for a, a0 in enumerate(a_list):
                Na = a0 * I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalrustImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if e is not None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate(
                        [e['dtScores'][0:maxDet] for e in E])

                    # different sorting method generates slightly different
                    # results. mergesort is used to be consistent as Matlab
                    # implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]

                    dtm = np.concatenate(
                        [e['dtMatches'][:, 0:maxDet] for e in E], axis=1)[:,
                                                                          inds]
                    dtIg = np.concatenate(
                        [e['dtIgnore'][:, 0:maxDet] for e in E], axis=1)[:,
                                                                         inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg == 0)
                    if npig == 0:
                        continue
                    tps = np.logical_and(dtm, np.logical_not(dtIg))
                    fps = np.logical_and(np.logical_not(dtm),
                                         np.logical_not(dtIg))

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp + tp + np.spacing(1))
                        q = np.zeros((R, ))
                        ss = np.zeros((R, ))

                        if nd:
                            recall_rst[t, k, a, m] = rc[-1]
                        else:
                            recall_rst[t, k, a, m] = 0

                        # numpy is slow without cython optimization for
                        # accessing elements use python array gets significant
                        # speed improvement
                        pr = pr.tolist()
                        q = q.tolist()

                        for i in range(nd - 1, 0, -1):
                            if pr[i] > pr[i - 1]:
                                pr[i - 1] = pr[i]

                        inds = np.searchsorted(rc, p.recThrs, side='left')
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except:  # noqa: E722
                            pass
                        precision_rst[t, :, k, a, m] = np.array(q)
                        scores_rst[t, :, k, a, m] = np.array(ss)

        for k, k0 in enumerate(ksta_list):
            Nk = k0 * A0 * I0
            for a, a0 in enumerate(a_list):
                Na = a0 * I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalstateImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if e is not None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate(
                        [e['dtScores'][0:maxDet] for e in E])

                    # different sorting method generates slightly different
                    # results. mergesort is used to be consistent as Matlab
                    # implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]

                    dtm = np.concatenate(
                        [e['dtMatches'][:, 0:maxDet] for e in E], axis=1)[:,
                                                                          inds]
                    dtIg = np.concatenate(
                        [e['dtIgnore'][:, 0:maxDet] for e in E], axis=1)[:,
                                                                         inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg == 0)
                    if npig == 0:
                        continue
                    tps = np.logical_and(dtm, np.logical_not(dtIg))
                    fps = np.logical_and(np.logical_not(dtm),
                                         np.logical_not(dtIg))

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp + tp + np.spacing(1))
                        q = np.zeros((R, ))
                        ss = np.zeros((R, ))

                        if nd:
                            recall_sta[t, k, a, m] = rc[-1]
                        else:
                            recall_sta[t, k, a, m] = 0

                        # numpy is slow without cython optimization for
                        # accessing elements use python array gets significant
                        # speed improvement
                        pr = pr.tolist()
                        q = q.tolist()

                        for i in range(nd - 1, 0, -1):
                            if pr[i] > pr[i - 1]:
                                pr[i - 1] = pr[i]

                        inds = np.searchsorted(rc, p.recThrs, side='left')
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except:  # noqa: E722
                            pass
                        precision_sta[t, :, k, a, m] = np.array(q)
                        scores_sta[t, :, k, a, m] = np.array(ss)

        self.eval = {
            'params': p,
            'counts': [T, R, Kcls, Krst, Ksta, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision_cls': precision_cls,
            'precision_rst': precision_rst,
            'precision_sta': precision_sta,
            'recall_cls': recall_cls,
            'recall_rst': recall_rst,
            'recall_sta': recall_sta,
            'scores_cls': scores_cls,
            'scores_rst': scores_rst,
            'scores_sta': scores_sta,
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc - tic))

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter
        setting
        '''
        def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
            p = self.params
            iStr = '{} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f} | {:0.3f} | {:0.3f}'  # noqa: E501
            #titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap == 1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [
                i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng
            ]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s_cls = self.eval['precision_cls']
                s_rst = self.eval['precision_rst']
                s_sta = self.eval['precision_sta']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s_cls = s_cls[t]
                    s_rst = s_rst[t]
                    s_sta = s_sta[t]
                s_cls = s_cls[:, :, :, aind, mind]
                s_rst = s_rst[:, :, :, aind, mind]
                s_sta = s_sta[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s_cls = self.eval['recall_cls']
                s_rst = self.eval['recall_rst']
                s_sta = self.eval['recall_sta']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s_cls = s_cls[t]
                    s_rst = s_rst[t]
                    s_sta = s_sta[t]
                s_cls = s_cls[:, :, aind, mind]
                s_rst = s_rst[:, :, aind, mind]
                s_sta = s_sta[:, :, aind, mind]
            #print(s_cls)
            if len(s_cls[s_cls > -1]) == 0:
                 mean_s_cls = -1
            else:
                 mean_s_cls = np.mean(s_cls[s_cls > -1])

            if len(s_rst[s_rst > -1]) == 0:
                 mean_s_rst = -1
            else:
                 mean_s_rst = np.mean(s_rst[s_rst > -1])

            if len(s_sta[s_sta > -1]) == 0:
                 mean_s_sta = -1
            else:
                 mean_s_sta = np.mean(s_sta[s_sta > -1])

            print(
                iStr.format(typeStr, iouStr, areaRng, maxDets,
                            mean_s_cls, mean_s_rst, mean_s_sta))
            return [mean_s_cls, mean_s_rst, mean_s_sta]

        def _summarizeDets():
            stats = np.zeros((12, 3))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1,
                                  iouThr=.75,
                                  maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1,
                                  areaRng='small',
                                  maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1,
                                  areaRng='medium',
                                  maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1,
                                  areaRng='large',
                                  maxDets=self.params.maxDets[2])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0,
                                  areaRng='small',
                                  maxDets=self.params.maxDets[2])
            stats[10] = _summarize(0,
                                   areaRng='medium',
                                   maxDets=self.params.maxDets[2])
            stats[11] = _summarize(0,
                                   areaRng='large',
                                   maxDets=self.params.maxDets[2])
            return stats

        def _summarizeKps():
            stats = np.zeros((10, ))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats

        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize()

    def __str__(self):
        self.summarize()


class Params:
    '''
    Params for coco evaluation api
    '''
    def setDetParams(self):
        self.imgIds = []
        self.catclassIds = []
        self.catrustIds = []
        self.catstateIds = []
        # np.arange causes trouble.  the data point on arange is slightly
        # larger than the true value
        self.iouThrs = np.linspace(.5,
                                   0.95,
                                   int(np.round((0.95 - .5) / .05)) + 1,
                                   endpoint=True)
        self.recThrs = np.linspace(.0,
                                   1.00,
                                   int(np.round((1.00 - .0) / .01)) + 1,
                                   endpoint=True)
        self.maxDets = [1, 10, 100]
        self.areaRng = [[0**2, 1e5**2], [0**2, 32**2], [32**2, 96**2],
                        [96**2, 1e5**2]]
        self.areaRngLbl = ['all', 'small', 'medium', 'large']
        self.useCats = 1

    def setKpParams(self):
        self.imgIds = []
        self.catclassIds = []
        self.catrustIds = []
        self.catstateIds = []
        # np.arange causes trouble.  the data point on arange is slightly
        # larger than the true value
        self.iouThrs = np.linspace(.5,
                                   0.95,
                                   int(np.round((0.95 - .5) / .05)) + 1,
                                   endpoint=True)
        self.recThrs = np.linspace(.0,
                                   1.00,
                                   int(np.round((1.00 - .0) / .01)) + 1,
                                   endpoint=True)
        self.maxDets = [20]
        self.areaRng = [[0**2, 1e5**2], [32**2, 96**2], [96**2, 1e5**2]]
        self.areaRngLbl = ['all', 'medium', 'large']
        self.useCats = 1
        self.kpt_oks_sigmas = np.array([
            .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07,
            .87, .87, .89, .89
        ]) / 10.0

    def __init__(self, iouType='segm'):
        if iouType == 'segm' or iouType == 'bbox':
            self.setDetParams()
        elif iouType == 'keypoints':
            self.setKpParams()
        else:
            raise Exception('iouType not supported')
        self.iouType = iouType
        # useSegm is deprecated
        self.useSegm = None
