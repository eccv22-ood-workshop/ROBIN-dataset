# ROBIN-dataset

This is the official repository for the [ROBIN](https://arxiv.org/abs/2111.14341) dataset.

You can access the dataset from [here](https://drive.google.com/drive/folders/1nJo60wphQ36T_E-iAuhN2ftLYh2285xb?usp=sharing)

The `.csv` file in each folder of the zip file contains the bounding box and 3D pose annotations for each images, please refer to this [repo](https://github.com/YoungXIAO13/PoseContrast/blob/main/data/Pascal3D/create_annotation.py) to see how we convert the matlab annotations into these csv files.

For image classification, you will need to crop the bounding boxes with 10 pixel padding from the images.

For object detection, we recommand using the [pascal-voc-writer](https://github.com/AndrewCarterUK/pascal-voc-writer) library to convert the `.csv` files into PASCAL-VOC format for training and testing.

For 3D pose estimation, we recommand using the [NeMo](https://github.com/Angtian/NeMo) pipeline to train and test the model.

We will also provide the data processing scripts, baselines, and add the occlusion nuisance shortly.

We are still waiting for a confirmation of evaluation metrics from our sponsor, so the final evaluation method is not confirmed and the codalab server is not online.
But please be noted, we will be restricting the performance on the IID test set in our dataset, for example, if the performance of your model on the IID test set three sigmas away from the mean value of our baseline, your submission will not be valid.


## Q&A

Q: Is it allowed to use additional data for training purposes?

A: **No**, only ImageNet-1k and ImageNet-1k pretrained models can be used for the challenge.

Q: Can we use strong data augmentations (e.g. GANs), ensembling, or test time training techniques?

A: **Yes**, but please note that we penalize submissions that are too far away on the IID test performance, and most of these techniques also improve the IID performance, so use them with caution.






