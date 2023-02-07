# OOD-CV dataset

## Dataset download

Please access the data from [here](https://bzhao.me/OOD-CV/)

## Phase-2 dataset

The dataset used by Phase-2 can be accessed from [here](https://drive.google.com/file/d/1xOxlrTjQb4V2uZFrp1LUdJniUI_ut_gB/view?usp=sharing), an email describing the rules for Phase-2 and final prize consideration will be send out after the phase-2 begins.

### Rules about phase-1 data

1. The phase-1 data can be used as a validation set, but the labels cannot be used for training.
2. The phase-1 data also can not be used as an unlabeled set for training for phase-2 submissions


## Evaluation

Our aim is to measure model robustness w.r.t. OOD nuisance factors. Therefore, the final benchmark scoring is data and accuracy constrained. This means, that to be valid a submission must:
1) Only use the training data that we provide. Using outside data is not allowed.
2) The model’s accuracy on the I.I.D. test set must be lower than a pre-defined threshold (which is defined by the performance of a baseline model).
The final benchmark score is then measured as average performance on the held-out O.O.D. test set.

The I.I.D. accuracy thresholds are as follows:
Image Classification = 91.1 [top-1 accuracy]
Object Detection = 79.9 [mAP@50]
Pose Estimation = 68.7 [Acc@pi/6]
Each accuracy threshold was determined by training the baseline models five times, followed by computing the mean performance and adding three standard deviations.

The evaluation code used on the CodaLab server is provided in the `evaluation` folder.

## CodaLab Servers

| Tasks                | CodaLab                                            |
|----------------------|----------------------------------------------------|
| Image-Classification | https://codalab.lisn.upsaclay.fr/competitions/6781 |
| Object-Detection     | https://codalab.lisn.upsaclay.fr/competitions/6784 |
| 3D-Pose-Estimation   | https://codalab.lisn.upsaclay.fr/competitions/6783 |


## Changes

1. The Phase-1 of the competition will not be a code submission challenge, we have released all the test data and labels in this repo. And Phase-1 will last longer than original planed, we will ask each team to provide a description of their developing environment at the end of Phase-1, Phase-2 will still be code submission challenge.
2. We will be using Top-1 accuracy for image-classification, mAP@50 for object detection, and Acc@pi/6 for pose estimation as the metric, the IID test performance will also be considered as per request of the sponsor, we will penalize submissions that are significantly different in IID performance with our baseline.
3. The only limitation now is that the model should only be trained on the given training set and/or the ImageNet-1k dataset, no additional dataset is allowed. You can use any ensemble, data augmentation, or test-time training techniques.

---

This is the official repository for the [OOD-CV](https://arxiv.org/abs/2111.14341) dataset.

The `.csv` file in each folder of the zip file contains the bounding box and 3D pose annotations for each images, please refer to this [repo](https://github.com/YoungXIAO13/PoseContrast/blob/main/data/Pascal3D/create_annotation.py) to see how we convert the matlab annotations into these csv files. Please note that for nuisance, there is no images for some particular categories, e.g. there is no diningtables in an OOD weather.

For image classification, you will need to crop the bounding boxes with 10 pixel padding from the images.

For object detection, we recommand using the [pascal-voc-writer](https://github.com/AndrewCarterUK/pascal-voc-writer) library to convert the `.csv` files into PASCAL-VOC format for training and testing.

For 3D pose estimation, we recommand using the [NeMo](https://github.com/Angtian/NeMo) pipeline to train and test the model.

We will also provide the data processing scripts, baselines, and add the occlusion nuisance shortly.


## Q&A

Q: Is it allowed to use additional data for training purposes?

A: **No**, for the data, only the images and the classification labels from ImageNet-1k can be used. For pretrained models, only ImageNet-1k pretrained models can be used for the challenge.

Q: Can we use strong data augmentations (e.g. GANs), ensembling, or test time training techniques?

A: **Yes**, but please note that we penalize submissions that are too far away on the IID test performance, and most of these techniques also improve the IID performance, so use them with caution.






