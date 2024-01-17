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
    """
    Main function for preprocessing the Waterbirds dataset

    Loads the dataset, applies transformations, and saves the processed data
    
    Args:
    args: Command line arguments provided to the script
    """
    # Read the config file 
    # (stores attributes like path to dataset or transform size)
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
    """
    Parses command line arguments and initiates the data preprocessing.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Root path of the Waterbirds dataset")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], required=True, help="Dataset split to process")
    parser.add_argument("--resize_size", type=int, default=224, help="Size to resize images")
    parser.add_argument("--save_path", type=str, default="processed/", help="Path to save the processed dataset")

    args = parser.parse_args()
    preprocess_waterbirds(args)

if __name__ == "__main__":
    main()
