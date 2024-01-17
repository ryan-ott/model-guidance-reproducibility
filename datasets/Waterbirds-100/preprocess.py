import argparse
import torch
import os
import json 
from tqdm import tqdm
import torchvision.transforms as transforms
from waterbirds_new import Waterbirds


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def dict_to_obj(d):
    if isinstance(d, dict):
        return Struct(**{k: dict_to_obj(v) for k, v in d.items()})
    return d


def preprocess_waterbirds(args):
    # Read the config file
    with open('config.json', 'r') as f:
        cfg_dict = json.load(f)

    cfg = dict_to_obj(cfg_dict)

    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize((args.resize_size, args.resize_size)),
        transforms.ToTensor()
    ])

    # Initialize Waterbirds dataset
    dataset = Waterbirds(root=args.data_root, cfg=cfg, split=args.split, transform=transform)

    # Prepare storage for processed data
    processed_data = []
    
    # Iterate over the dataset and process
    for idx in tqdm(range(len(dataset))):
        data = dataset[idx]
        processed_data.append(data)

    # Save the processed data
    os.makedirs(args.save_path, exist_ok=True)
    torch.save(processed_data, os.path.join(args.save_path, f"waterbirds_{args.split}_processed.pt"))

    print(f"Processed data saved to {os.path.join(args.save_path, f'waterbirds_{args.split}_processed.pt')}")

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Root path of the Waterbirds dataset")
    # parser.add_argument("--cfg", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], required=True, help="Dataset split to process")
    parser.add_argument("--resize_size", type=int, default=224, help="Size to resize images")
    parser.add_argument("--save_path", type=str, default="processed/", help="Path to save the processed dataset")

    args = parser.parse_args()
    preprocess_waterbirds(args)

if __name__ == "__main__":
    main()
