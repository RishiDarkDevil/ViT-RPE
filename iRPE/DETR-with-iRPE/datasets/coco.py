# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path
from typing import List, Any

import torch
import torch.utils.data
import torchvision
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
from PIL import Image
from io import BytesIO
import os
import zipfile

import datasets.transforms as T

from torchvision.datasets import VisionDataset
from typing import Any, Callable, Optional, Tuple, List

ZIPS = dict()

def get_zip_handle(fname):
    global ZIPS
    if fname not in ZIPS:
        handle = zipfile.ZipFile(fname, 'r')
        ZIPS[fname] = handle
    return ZIPS[fname]


READ_IMAGE_IF_EXISTED = True
def my_open(root, fname):
    global READ_IMAGE_IF_EXISTED
    '''
    root:  xxx/train
    fname: file
    '''
    root = str(root)
    if READ_IMAGE_IF_EXISTED:
        image_fname = os.path.join(root, fname)
        try:
            return open(image_fname, 'rb').read()
        except:
            # switch to reading zip file because image file not found
            READ_IMAGE_IF_EXISTED = False
    zip_fname = root + '.zip'
    handle = get_zip_handle(zip_fname)
    base_name = os.path.basename(root)
    zname = f'{base_name}/{fname}'
    return handle.read(zname)


def my_Image_open(root, fname):
    '''
    root:  xxx/train2017
    fname: file
    '''
    iob = BytesIO(my_open(root, fname))
    return Image.open(iob)

class CocoDetectionOptim(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annot (string): Path to json ** annotations directory **.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        # annFile: str,
        annot: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        # from pycocotools.coco import COCO

        # self.coco = COCO(annFile)
        self.ann_paths = [os.path.join(annot, f) for f in os.listdir(annot) if f.endswith('.json')]
        # self.ids = list(sorted(self.coco.imgs.keys()))

    def _load_image(self, index: int) -> Image.Image:
        coco = COCO(self.ann_paths[index])
        id = list(coco.imgs.keys())[0]
        path = coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, index: int) -> List[Any]:
        coco = COCO(self.ann_paths[index])
        id = list(coco.imgs.keys())[0]
        return coco.loadAnns(coco.getAnnIds(id))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        # id = self.ids[index]
        image = self._load_image(index)
        target = self._load_target(index)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.ann_paths)


class CocoDetection(CocoDetectionOptim):
    def __init__(self, img_folder, ann_folder, transforms, return_masks):
        super(CocoDetection, self).__init__(img_folder, ann_folder)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img = self._load_image(idx)
        target = self._load_target(idx)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        coco = COCO(self.ann_paths[idx])
        image_id = list(coco.imgs.keys())[0]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

    def _load_image(self, index: int) -> Image.Image:
        coco = COCO(self.ann_paths[index])
        id = list(coco.imgs.keys())[0]
        path = coco.loadImgs(id)[0]["file_name"]
        return my_Image_open(self.root, path).convert('RGB')

    def _load_target(self, index) -> List[Any]:
        coco = COCO(self.ann_paths[index])
        id = list(coco.imgs.keys())[0]
        return coco.loadAnns(coco.getAnnIds(id))


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    PATHS = {
        "train": (root / "train", root / "annotations"),
        "val": (root / "val", root / "annotations"),
    }

    img_folder, ann_folder = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_folder, transforms=make_coco_transforms(image_set), return_masks=args.masks)
    return dataset
