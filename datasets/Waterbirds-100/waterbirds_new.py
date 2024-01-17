import torch
from torchvision import transforms
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
from PIL import Image

GROUP_NAMES = np.array(['Land_on_Land', 'Land_on_Water', 'Water_on_Land', 'Water_on_Water'])

def get_label_mapping():
    return np.array(['Landbird', 'Waterbird'])

# Other constants and helper functions remain unchanged

class Waterbirds(torch.utils.data.Dataset):
    def __init__(self, root, cfg, split='train', transform=None):
        self.cfg = cfg
        self.original_root = os.path.expanduser(root)
        self.transform = transform
        self.split = split
        self.root = os.path.join(self.original_root, cfg.DATA.WATERBIRDS_DIR)
        self.return_bbox = True
        self.size = cfg.DATA.SIZE

        print('WATERBIRDS DIR: {}'.format(self.root))

        # metadata
        self.metadata_df = pd.read_csv(os.path.join(self.root, 'metadata.csv'))

        # Get the y values
        self.labels = self.metadata_df['y'].values
        self.num_classes = 2

        self.confounder_array = self.metadata_df['place'].values
        self.n_confounders = 1
        self.n_groups = pow(2, 2)
        self.group_array = (self.labels*(self.n_groups/2) + self.confounder_array).astype('int')

        # Extract filenames and splits
        self.filename_array = self.metadata_df['img_filename'].values
        self.split_array = self.metadata_df['split'].values
        self.split_dict = {'train': 0, 'val': 1, 'test': 2}

        self.data = np.array([os.path.join(self.root, filename) for filename in self.filename_array])

        mask = self.split_array == self.split_dict[self.split]
        num_split = np.sum(mask)
        self.indices = np.where(mask)[0]

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
            bboxes = pd.read_csv(os.path.join(self.original_root, 'CUB_200_2011', 'bounding_boxes.txt'), header=None)
            self.bbox_coords = np.zeros((self.data.shape[0], 4))
            for i, row in enumerate(bboxes.values):
                coords = row[0].split(' ')[1:]
                coords = np.array(coords).astype(float)
                self.bbox_coords[i] = coords

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

# Other functions like get_loss_upweights remain unchanged
