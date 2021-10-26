"""Visualizes CNN activation maps to see where the CNN focuses on to extract features.

Reference:
    - Zagoruyko and Komodakis. Paying more attention to attention: Improving the
      performance of convolutional neural networks via attention transfer. ICLR, 2017
    - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
"""
import numpy as np
import os.path as osp
import argparse
import cv2
import torch
from mmcv.image import tensor2imgs
from torch.nn import functional as F
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
GRID_SPACING = 10


@torch.no_grad()
def visactmap(
    model,
    test_loader,
    save_dir,
    width,
    height,
    use_gpu,
    img_mean=None,
    img_std=None
):
    model.eval()
    actmap_dir = osp.join('actmap', save_dir)
    for i, data in enumerate(test_loader):
        img_tensor = data['img'][0]
        img_metas = data['img_metas'][0].data[0]

        if img_mean is None or img_std is None:
            # use imagenet mean and std
            img_mean = img_metas[0]['img_norm_cfg']['mean']
            img_std = img_metas[0]['img_norm_cfg']['std']
        paths = img_metas[0]['ori_filename']

        if use_gpu:
            imgs = img_tensor.cuda()

            # forward to get convolutional feature maps
        try:
            with torch.no_grad():
                model = MMDataParallel(model, device_ids=[0])
                outputs = model(img_tensor)[-1]
                print(outputs.size())
        except TypeError:
            raise TypeError(
                'forward() got unexpected keyword argument "return_featuremaps". '
                'Please add return_featuremaps as an input argument to forward(). When '
                'return_featuremaps=True, return feature maps only.'
            )

        if outputs.dim() != 4:
            raise ValueError(
                'The model output is supposed to have '
                'shape of (b, c, h, w), i.e. 4 dimensions, but got {} dimensions. '
                'Please make sure you set the model output at eval mode '
                'to be the last convolutional feature maps'.format(
                    outputs.dim()
                )
            )

            # compute activation maps
        outputs = (outputs**2).sum(1)
        b, h, w = outputs.size()
        outputs = outputs.view(b, h * w)
        outputs = F.normalize(outputs, p=2, dim=1)
        outputs = outputs.view(b, h, w)

        if use_gpu:
            imgs, outputs = imgs.cpu(), outputs.cpu()

        for j in range(outputs.size(0)):
            # get image name
            imname = osp.basename(osp.splitext(paths)[0])
            # RGB image
            img = imgs[j, ...]
            _, height, width = img.shape
            for t, m, s in zip(img, img_mean, img_std):
                t.mul_(s).add_(m).clamp_(0, 1)
            img_np = np.uint8(np.floor(img.numpy() * 255))
            img_np = img_np.transpose((1, 2, 0)) # (c, h, w) -> (h, w, c)

            # activation map
            am = outputs[j, ...].numpy()
            am = cv2.resize(am, (width, height))
            am = 255 * (am - np.min(am)) / (
                np.max(am) - np.min(am) + 1e-12
            )
            am = np.uint8(np.floor(am))
            am = cv2.applyColorMap(am, cv2.COLORMAP_JET)

            # overlapped
            overlapped = img_np*0.3 + am*0.7
            overlapped[overlapped > 255] = 255
            overlapped = overlapped.astype(np.uint8)

            # save images in a single figure (add white spacing between images)
            # from left to right: original image, activation map, overlapped image
            grid_img = 255 * np.ones(
                (height, 3*width + 2*GRID_SPACING, 3), dtype=np.uint8
            )
            grid_img[:, :width, :] = img_np[:, :, ::-1]
            grid_img[:,
                     width + GRID_SPACING:2*width + GRID_SPACING, :] = am
            grid_img[:, 2*width + 2*GRID_SPACING:, :] = overlapped
            cv2.imwrite(osp.join(actmap_dir, imname + '.jpg'), grid_img)

        if (i+1) % 10 == 0:
            print(
                '- done batch {}/{}'.format(
                    i + 1, len(test_loader)
                )
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--save-dir', type=str, default='log')
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=128)
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True

    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    if samples_per_gpu > 1:
        # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    print(model.CLASSES)
    weight_dict = torch.load(args.checkpoint)
    model.load_state_dict(weight_dict['state_dict'])
    if use_gpu:
        model = model.cuda()
    visactmap(
        model.backbone, data_loader, args.save_dir, args.width, args.height, use_gpu
    )


if __name__ == '__main__':
    main()
