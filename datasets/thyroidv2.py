import os
import numpy as np
import torch
from PIL import Image
from pathlib import Path
import cv2
import json

import functools
print = functools.partial(print, flush=True)

import datasets.transforms as T

def make_transforms(image_set):
    trans_list = []
    trans_list.append(T.PILToTensor())
    trans_list.append(T.ToDtype(torch.float))
    if image_set == "train":
        trans_list.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(trans_list)

ASSUME_VALID = True # if True, we assume all frames are valid
SHAPE_TYPE = 0
BBOX_TYPE = 1

class ThyroidV2Dataset(torch.utils.data.Dataset):
    def __init__(self, video_folder, anno_folder, transforms, image_set):
        self.video_folder = video_folder
        self.anno_folder = anno_folder
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.video_paths = list(sorted(os.listdir(video_folder)))
        # exlude the directory of cache
        self.video_paths = [x for x in self.video_paths if x != 'cache']
        self.annotation_paths = list(sorted(os.listdir(anno_folder)))
        assert len(self.video_paths) == len(self.annotation_paths)
        self.cache_dir = os.path.join(video_folder, 'cache')
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        self.annotations = []

        self.total_frames = 0
        # valid = have annotations + valid frames

        self.total_vaild_frames = 0
        self.vaild_frames_video_id = []
        self.vaild_frames_frame_id = []
        self.vaild_frames_list_id = []
        self.vaild_frames_key_type = [] # 0: shape, 1: bbox

        for i in range(len(self.video_paths)):
            video_path = os.path.join(video_folder, self.video_paths[i])
            print(f"parse video {i}/{len(self.video_paths)}: {video_path}...")
            anno_path = os.path.join(anno_folder, self.annotation_paths[i])

            video = cv2.VideoCapture(video_path)
            with open(anno_path, 'r') as f:
                anno = json.load(f)

            self.annotations.append(anno)

            self.height, self.width = anno['FileInfo']['Height'], anno['FileInfo']['Width']
            if self.width != int(video.get(cv2.CAP_PROP_FRAME_WIDTH)):
                import pdb; pdb.set_trace()
            assert self.width == int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            assert self.height == int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

            self.total_frames += num_frames

            # refer to http://1.116.185.48
            label_model = anno['Models']['ColorLabelTableModel']
            # [{'Color': [255, 255, 0, 255], 'Desc': 'RLN', 'ID': 1}, {'Color': [85, 255, 255, 255], 'Desc': 'PTG', 'ID': 2}]
            self.target_label_type = None
            for label in label_model:
                if label['Desc'] == 'PTG':
                    self.target_label_type = label['ID']
                    break
            assert self.target_label_type is not None

            shapes = anno['Polys'][0]['Shapes']
            bbox_model = anno['Models']['BoundingBoxLabelModel']

            if shapes:
                for shape_id, shape in enumerate(shapes):
                    # print(f"parse video {i}/{len(self.video_paths)}, shape {shape_id}/{len(shapes)}...")
                    frame_idx = shape['ImageFrame']
                    label_type = shape['labelType']
                    points = shape['Points']
                    if label_type != self.target_label_type:
                        continue
                    assert len(points) > 0
                    if not ASSUME_VALID:
                        video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret, _ = video.read() # ret = 1 if the frame is valid
                        if ret:
                            self.total_vaild_frames += 1
                            self.vaild_frames_video_id.append(i)
                            self.vaild_frames_frame_id.append(frame_idx)
                            self.vaild_frames_list_id.append(shape_id)
                            self.vaild_frames_key_type.append(SHAPE_TYPE)
                    else:
                        self.total_vaild_frames += 1
                        self.vaild_frames_video_id.append(i)
                        self.vaild_frames_frame_id.append(frame_idx)
                        self.vaild_frames_list_id.append(shape_id)
                        self.vaild_frames_key_type.append(SHAPE_TYPE)

            if bbox_model:
                for bbox_id, bbox in enumerate(bbox_model):
                    # print(f"parse video {i}/{len(self.video_paths)}, bbox {bbox_id}/{len(bbox_model)}...")
                    frame_idx = bbox['FrameCount']
                    label_type = bbox['Label']
                    if label_type != self.target_label_type:
                        continue
                    xmin, ymin, _ = bbox["p1"]
                    xmax, ymax, _ = bbox["p2"]
                    if xmin == 0 and ymin == 0 and xmax == 0 and ymax == 0:
                        # odd output from Pair
                        continue
                    if xmin >= xmax or ymin >= ymax:
                        continue
                    if not ASSUME_VALID:
                        video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret, _ = video.read() # ret = 1 if the frame is valid
                        if ret:
                            self.total_vaild_frames += 1
                            self.vaild_frames_video_id.append(i)
                            self.vaild_frames_frame_id.append(frame_idx)
                            self.vaild_frames_list_id.append(bbox_id)
                            self.vaild_frames_key_type.append(BBOX_TYPE)
                    else:
                        self.total_vaild_frames += 1
                        self.vaild_frames_video_id.append(i)
                        self.vaild_frames_frame_id.append(frame_idx)
                        self.vaild_frames_list_id.append(bbox_id)
                        self.vaild_frames_key_type.append(BBOX_TYPE)

        self.vaild_frames_video_id = np.array(self.vaild_frames_video_id)
        self.vaild_frames_frame_id = np.array(self.vaild_frames_frame_id)
        self.vaild_frames_list_id = np.array(self.vaild_frames_list_id)
        self.vaild_frames_key_type = np.array(self.vaild_frames_key_type)

    def __getitem__(self, idx):
        video_path = os.path.join(self.video_folder, self.video_paths[self.vaild_frames_video_id[idx]])
        anno_id = self.vaild_frames_video_id[idx]
        frame_id = self.vaild_frames_frame_id[idx]
        list_id = self.vaild_frames_list_id[idx]
        key_type = self.vaild_frames_key_type[idx]

        cache_path = os.path.join(self.cache_dir, os.path.basename(video_path).replace('.mp4', f'_{frame_id}.jpg'))
        if not os.path.exists(cache_path):
            video = cv2.VideoCapture(video_path)
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = video.read() # ret = 1 if the frame is valid, img: (H, W, 3)
            if ret == 0:
                # a walkaround for invalid frames
                print(f"Warning: video {video_path}, frame {frame_id} is invalid, use the first frame instead.")
                return self.__getitem__(0)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            # save to cache
            img.save(cache_path)
        else:
            img = Image.open(cache_path)

        anno = self.annotations[anno_id]
        if key_type == SHAPE_TYPE:
            shape = anno['Polys'][0]['Shapes'][list_id]
            frame_idx = shape['ImageFrame']
            label_type = shape['labelType']
            assert frame_idx == frame_id
            assert label_type == self.target_label_type
            points = shape['Points']
            xx = np.array([point['Pos'][0] for point in points])
            yy = np.array([point['Pos'][1] for point in points])
            xmin, xmax = np.min(xx), np.max(xx)
            ymin, ymax = np.min(yy), np.max(yy)
            boxes = [[xmin, ymin, xmax, ymax]]
        else:
            bbox_model = anno['Models']['BoundingBoxLabelModel']
            bbox = bbox_model[list_id]
            frame_idx = bbox['FrameCount']
            assert frame_idx == frame_id
            label_type = bbox['Label']
            assert label_type == self.target_label_type
            xmin, ymin, _ = bbox["p1"]
            xmax, ymax, _ = bbox["p2"]
            boxes = [[xmin, ymin, xmax, ymax]]

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # there is only one class
        num_objs = 1
        labels = torch.ones((num_objs,), dtype=torch.int64)

        # bug fixing: `AssertionError: Results do not correspond to current coco set`
        # https://stackoverflow.com/questions/76798069/assertionerror-results-do-not-correspond-to-current-coco-set-wrong-types-and
        image_id = idx
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
        return self.total_vaild_frames

def build(image_set, args):
    root = Path(args.thyroid_path)
    assert root.exists(), f'provided Thyroid path {root} does not exist'
    PATHS = {
        "train": (root / "train" / "videos", root / "train" / "annotations"),
        "val": (root / "val" / "videos", root / "val" / "annotations"),
        "test": (root / "test" / "videos", root / "test" / "annotations"),
    }

    if args.eval and args.test:
        print('Inference on test-dev.')
        image_set = 'test'
    video_folder, anno_folder = PATHS[image_set]
    dataset = ThyroidV2Dataset(video_folder, anno_folder, transforms=make_transforms(image_set), image_set=image_set)
    return dataset
