import argparse
import torch
import os
import json 
from tqdm import tqdm
import torchvision.transforms as transforms
from waterbirds_new import Waterbirds

# This class and the following function convert
# a dict into a Struct object; necessary for
# parsing the config.json file
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def dict_to_obj(d):
    if isinstance(d, dict):
        return Struct(**{k: dict_to_obj(v) for k, v in d.items()})
    return d


def preprocess_waterbirds(args):
    # Read the config file
    with open('./datasets/Waterbirds-100/config.json', 'r') as f:
        cfg_dict = json.load(f)

    cfg = dict_to_obj(cfg_dict)

    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize((args.resize_size, args.resize_size)),
        transforms.ToTensor()
    ])

    # Initialize Waterbirds dataset
    dataset = Waterbirds(root=args.data_root, task=args.task, cfg=cfg, split=args.split, transform=transform)

    # Turn on the filter if working on validation set
    apply_val_filter = args.split == 'val'

    processed_data = []    
    # Iterate over the dataset and process
    for idx in tqdm(range(len(dataset))):
        data = dataset[idx]

        # Filter out unwanted groups for validation set
        if apply_val_filter and data['group'].item() in [1, 2]:
            continue

        processed_data.append(data)

    # Save the processed data
    save_path = args.task + '_processed/'
    os.makedirs(save_path, exist_ok=True)
    torch.save(processed_data, os.path.join(save_path, f"{args.split}.pt"))

    print(f"Processed data saved to {os.path.join(save_path, f'{args.split}.pt')}")

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Root path of the Waterbirds dataset")
    parser.add_argument("--task", type=str, choices=["birds", "background"], required=True, help="Whether to classify birds or background")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], required=True, help="Dataset split to process")
    parser.add_argument("--resize_size", type=int, default=224, help="Size to resize images")

    args = parser.parse_args()
    preprocess_waterbirds(args)

if __name__ == "__main__":
    main()
