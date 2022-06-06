# ROBIN-dataset

This is the official repository for the Robustness to Individual Nuisances in Real-World Out-of-Distribution Shifts (ROBIN) dataset.


## Image-classification

In the `image-classification` folder we provide a baseline for training and validating a classification model on our dataset.

The dataset used for image classification is provided [here](https://drive.google.com/drive/folders/1f2Ch6X1qnI6-OWugESEmeysRxx4IVkTL?usp=sharing), where the `ROBIN-cls-train` folder is organized in a way that can be directly readable by the `ImageFolder` class in PyTorch, and the `ROBIN-cls-val` folder contains 5 subfolders that represents the 5 nuisances, `shape`, `pose`, `texture`, `context`, and `weather`, where each subfolder is readable by the `ImageFolder` class, the validation set only contains `200` images, where each classes have `4` images per nuisance, the full test set will be accessible on the Codalab (releasing soon).
The `image-classification/main.py` is a classification baseline that have the minimal change from the PyTorch imagenet example, check the modification by:
```bash
diff image-classification/main.py <(curl https://raw.githubusercontent.com/pytorch/examples/master/imagenet/main.py)
```

## Object-detection

TBD



