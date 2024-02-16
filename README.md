# Studying How to Efficiently and Effectively Guide Models with Explanations - A Reproducibility Study

# TMLR/ MLRC Submission

## Setup
### Prerequisites
All the required packages can be installed using conda with the provided [environment.yml](environment.yml) file. 

### Data, Weights, and Model Training
Please refer to the [original repositiory by Rao et al. (2023)](https://github.com/sukrutrao/Model-Guidance?tab=readme-ov-file) for information on how to download the datasets and weights, as well as how to train the models.

### Waterbirds-100
Code for the experiments on the Waterbirds-100 dataset is not included in the original repository by Rao et al. (2023). Since this dataset is fairly small, all the necessary files are included in this repository for convenience under `datasets/Waterbirds-100`. In order to perform pre-processing, follow the instructions in the [README.md](datasets/) file in the `datasets/` directory.

There is a separate training script `train_waterbirds.py` for fine-tuning baseline and guided models on this dataset accross the two different tasks (1) conventional and (2) reversed. The command for executing this script follows the same pattern as for the model's on other datasets, with a single additional flag for setting the task. Thus, to execute the script for the conventional task, you would use:

```
python train_waterbirds.py --dataset Waterbirds-100 --task birds ...
```

and for the reversed task:

```
python train_waterbirds.py --dataset Waterbirds-100 --task background ...
```

replacing `...` with other argument values, as when running `train.py`.
