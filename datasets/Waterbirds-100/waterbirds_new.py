import torch
from torchvision import transforms
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
from PIL import Image

GROUP_NAMES = np.array(['Landbird_on_Land', 'Landbird_on_Water', 'Waterbird_on_Land', 'Waterbird_on_Water'])

def get_label_mapping(task):
    if task == 'birds':
        return np.array(['Landbird', 'Waterbird'])
    elif task == 'background':
        return np.array(['Land', 'Water'])
    else:
        raise ValueError(f"Unrecognized task type specified: {task}. Expected 'background' or 'birds'.")


class Waterbirds(torch.utils.data.Dataset):
    def __init__(self, root, task, cfg, split='train', transform=None):
        self.cfg = cfg
        self.original_root = os.path.expanduser(root)
        self.task = task
        self.transform = transform
        self.split = split
        self.root = os.path.join(self.original_root, cfg.DATA.WATERBIRDS_DIR)
        self.return_bbox = True
        self.size = cfg.DATA.SIZE

        print('WATERBIRDS DIR: {}'.format(self.root))

        # metadata
        self.metadata_df = pd.read_csv(os.path.join(self.root, 'metadata.csv'))

        self.num_classes = 2

        self.confounder_array = self.metadata_df['place'].values
        self.n_confounders = 1
        self.n_groups = pow(2, 2)
        self.group_array = (self.metadata_df['y'].values*(self.n_groups/2) + self.confounder_array).astype('int')

        # Extract filenames and splits
        self.filename_array = self.metadata_df['img_filename'].values
        self.split_array = self.metadata_df['split'].values
        self.split_dict = {'train': 0, 'val': 1, 'test': 2}

        self.data = np.array([os.path.join(self.root, filename) for filename in self.filename_array])

        mask = self.split_array == self.split_dict[self.split]
        num_split = np.sum(mask)
        self.indices = np.where(mask)[0]

        # Assign labels based on task type
        if task == 'background':
            self.labels = [(group_label == 1 or group_label == 3) for group_label in self.group_array]
        elif task == 'birds':
            self.labels = self.metadata_df['y'].values

        self.labels = torch.Tensor(self.labels)
        self.group_array = torch.Tensor(self.group_array)

        self.image_filenames = []
        self.labels_split = []
        self.group_labels_split = []
        for idx in self.indices:
            self.image_filenames.append(self.data[idx])
            self.labels_split.append(self.labels[idx])
            self.group_labels_split.append(self.group_array[idx])
        self.image_filenames = np.array(self.image_filenames)
        self.labels_split = torch.Tensor(self.labels_split)
        self.group_labels_split = torch.Tensor(self.group_labels_split)

        if self.return_bbox:
            # Read bounding boxes into a dictionary
            bbox_df = pd.read_csv(os.path.join(self.original_root, 'CUB_200_2011', 'bounding_boxes.txt'), sep=" ", header=None, index_col=0)
            bbox_dict = bbox_df.to_dict(orient='index')

            # Map each image to its bounding box using metadata 'img_id'
            self.bbox_coords = np.zeros((len(self.data), 4))
            for i, img_id in enumerate(self.metadata_df['img_id']):
                if int(img_id) in bbox_dict:
                    coords = bbox_dict[int(img_id)]
                    self.bbox_coords[i] = np.array([coords[1], coords[2], coords[3], coords[4]])

    def __getitem__(self, index):
        index = self.indices[index]
        path = self.data[index]
        label = self.labels[index]
        group = self.group_array[index]
        group = torch.Tensor([group])

        img = Image.open(path).convert('RGB')

        if self.return_bbox:
            bbox = self.bbox_coords[index]
            bbox = np.round(bbox).astype(int)
            arr = np.array(img)
            bbox_im = np.zeros(arr.shape[:2])
            bbox_im[bbox[1]:(bbox[1] + bbox[3]), bbox[0]:(bbox[0] + bbox[2])] = 1
            bbox_im = torch.Tensor(bbox_im).unsqueeze(0).unsqueeze(0)
            bbox_im = F.interpolate(bbox_im, size=(self.size, self.size), mode='bilinear', align_corners=False)[0]
        else:
            bbox_im  = torch.Tensor([-1])  # Placeholder

        if self.transform is not None:
            img = self.transform(img)

        return {
            'image_path': path,
            'image': img,
            'label': label,
            'group': group,
            'bbox': bbox_im,
            'index': index,
            'split': self.split
        }

    def __len__(self):
        return len(self.indices)
