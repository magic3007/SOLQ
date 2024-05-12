import os
import numpy as np
import torch
from PIL import Image
from pathlib import Path

import datasets.transforms as T

def make_transforms(image_set):
    transforms = []
    transforms.append(T.ToTensor())
    transforms.append(T.ToDtype(torch.float))
    if image_set == "train":
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

class ThyroidDataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, mask_folder, transforms, image_set):
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(img_folder)))
        self.masks = list(sorted(os.listdir(mask_folder)))
        if image_set is not "train":
            self.indices = np.arange(len(self.imgs))
        else:
            # in train mode, preprocess the groundtruth before training.
            # 1. disard images without annotations
            # 2. disard images with zero bounding box areas
            indices = []
            for idx in range(len(self.imgs)):
                mask_path = os.path.join(self.mask_folder, self.masks[idx])
                mask = Image.open(mask_path)
                # convert the PIL Image into a numpy array
                mask = np.array(mask)
                # instances are encoded as different colors
                obj_ids = np.unique(mask)
                # first id is the background, so remove it
                obj_ids = obj_ids[1:]
                # split the color-encoded mask into a set
                # of binary masks
                masks = mask == obj_ids[:, None, None]

                # get bounding box coordinates for each mask
                num_objs = len(obj_ids)
                boxes = []
                for i in range(num_objs):
                    pos = np.nonzero(masks[i])
                    xmin = np.min(pos[1])
                    xmax = np.max(pos[1])
                    ymin = np.min(pos[0])
                    ymax = np.max(pos[0])
                    boxes.append([xmin, ymin, xmax, ymax])

                if len(boxes) > 0:
                    boxes = torch.as_tensor(boxes, dtype=torch.float32)
                else:
                    continue

                area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

                if len(area)==0 or area.min() == 0:
                    continue

                indices.append(idx)

            self.indices = np.array(indices)

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.img_folder, self.imgs[self.indices[idx]])
        mask_path = os.path.join(self.mask_folder, self.masks[self.indices[idx]])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.nonzero(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        # target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd


        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.indices)

def build(image_set, args):
    root = Path(args.thyroid_path)
    assert root.exists(), f'provided Thyroid path {root} does not exist'
    PATHS = {
        "train": (root / "train" / "frames", root / "train" / "masks"),
        "val": (root / "val" / "frames", root / "val" / "masks"),
        "test": (root / "test" / "frames", root / "test" / "masks"),
    }

    if args.eval and args.test:
        print('Inference on test-dev.')
        image_set = 'test'
    img_folder, mask_folder = PATHS[image_set]
    dataset = ThyroidDataset(img_folder, mask_folder, transforms=make_transforms(image_set), image_set=image_set)
    return dataset
