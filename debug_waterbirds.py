import torch

# Path to your 'train.pt' file
dataset_path = './datasets/Waterbirds-100/birds_processed/train.pt'

# Load the dataset
dataset = torch.load(dataset_path)

# Inspect the structure of the loaded dataset
print(f"Type of loaded dataset: {type(dataset)}")
if isinstance(dataset, dict):
    print("Keys available in the dataset:", list(dataset.keys()))
    # Check if 'data' key exists
    if 'data' in dataset:
        print("Dataset contains 'data' key.")
    else:
        print("Dataset does not contain 'data' key.")
elif isinstance(dataset, list):
    print(f"Dataset is a list with length {len(dataset)}.")
    # Optionally, inspect the first element to see its structure
    if len(dataset) > 0:
        print("First element type:", type(dataset[0]))
        print("First element content (if it's a dictionary, print its keys):", dataset[0].keys() if isinstance(dataset[0], dict) else "Not a dictionary")
else:
    print("Dataset is neither a list nor a dictionary.")
