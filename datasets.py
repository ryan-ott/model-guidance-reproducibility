import numpy as np
import os
import torch
import utils


class VOCDetectParsed(torch.utils.data.Dataset):
    """
    Dataset class for PASCAL VOC2007 detection dataset

    Attributes:
    data (tensor): Tensor containing image data
    labels (tensor): Tensor containing labels for each image
    bbs (list): List of bounding boxes for each image
    annotated_fraction (float): Fraction of the dataset to consider as annotated

    Methods:
    __init__: Initializes dataset object, loads the data, applies settings
    __getitem__: Returns a single item from the dataset
    __len__: Returns the length of the dataset
    load_data: Loads data, label, and bounding box for a specified image and class
    collate_fn: Custom collate function for data loading
    """
    def __init__(self, root, image_set, transform=None, annotated_fraction=1.0):
        """
        Initializes the dataset object

        Args:
        root (str): Root dir where the data is stored
        image_set (str): Whether to use 'train', 'val' or 'test' subset of the data
        transform (callable, optional): Function that determines what transformation to apply to the images
        annotated_fraction (float): Fraction of the dataset to consider as annotated
        """
        super().__init__()
        # Load dataset from specified file
        data_dict = torch.load(os.path.join(root, image_set + ".pt"))
        self.data = data_dict["data"]
        self.labels = data_dict["labels"]
        self.bbs = data_dict["bbs"]

        # Make sure length of data, labels, and bbs is matching
        assert len(self.data) == len(self.labels)
        assert len(self.data) == len(self.bbs)
        self.annotated_fraction = annotated_fraction
        
        # If only a fraction of data is annotated, randomly select that fraction
        if self.annotated_fraction < 1.0:
            annotated_indices = np.random.choice(len(self.bbs), size=int(
                self.annotated_fraction*len(self.bbs)), replace=False)
            # Set bounding boxes to None for unannotated data
            for bb_idx in range(len(self.bbs)):
                if bb_idx not in annotated_indices:
                    self.bbs[bb_idx] = None
        self.transform = transform

    def __getitem__(self, idx):
        """
        Return an item by index

        Args:
        idx (int): Index of the item

        Returns:
        tuple: (image, label, bounding box) where the image is the transformed image if transform is provided
        """
        if self.transform is not None:
            return self.transform(self.data[idx]), self.labels[idx], self.bbs[idx]
        return self.data[idx], self.labels[idx], self.bbs[idx]

    def load_data(self, idx, pred_class):
        """
        Load data, label, and bounding box for a specific image and predicted class

        Args:
        idx (int): Index of the image in the dataset
        pred_class (int): The class for which to filter the bounding box

        Returns:
        tuple: (image, label, filtered bounding box)
        """
        img, labels, bbs = self.__getitem__(idx)
        label = labels[pred_class]
        bb = utils.filter_bbs(bbs, pred_class)
        return img, label, bb

    def __len__(self):
        """
        Return the length of the dataset

        Returns:
        int: Number of items in the dataset
        """
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        """
        Function that combines a list of samples into a mini-batch

        Args:
        batch (list of tuples): a list where each tuple consists of (data, labels, bounding boxes) for a single sample

        Returns:
        tuple: A tuple containing 3 elements:
                - data (tensor): batch of images combined into a single tensor
                - labels (tensor): batch of labels combined into a single tensor
                - bbs (list of lists): a list where each sublist contains bounding boxes corresponding to each img in the batch
        """
        # Combine img tensors from each sample into a single batch tensor
        data = torch.stack([item[0] for item in batch])
        
        # Combine label tensors from each sample into a single batch tensor
        labels = torch.stack([item[1] for item in batch])

        # Combine the bounding boxes from each sample into a list of lists
        bbs = [item[2] for item in batch]

        return data, labels, bbs
