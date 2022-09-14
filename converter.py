import pandas as pd
import numpy as np
import os
import os.path as osp
import shutil
from PIL import Image
from pascal_voc_writer import Writer
from glob import glob
import ntpath
from tqdm.auto import trange

file_folders = [
    'train/',
    'iid_test/',
    'nuisances/context/',
    'nuisances/pose/',
    'nuisances/shape/',
    'nuisances/texture/',
    'nuisances/weather/',
    'nuisances/occlusion/',
]

cls_names = ['aeroplane',
 'bicycle',
 'boat',
 'bus',
 'car',
 'chair',
 'diningtable',
 'motorbike',
 'sofa',
 'train']



# create image classification and pose estimation dataset
os.makedirs('ROBINv1.1-cls-pose', exist_ok=True)

for folder in file_folders:
    df = pd.read_csv(glob(osp.join('ROBINv1.1/', folder, '*.csv'))[0])
    img_save_dir = osp.join('ROBINv1.1-cls-pose/', folder, 'Images')
    os.makedirs(img_save_dir, exist_ok=True)
    for cls_name in cls_names:
        os.makedirs(osp.join('ROBINv1.1-cls-pose/', folder, 'Images', cls_name), exist_ok=True)
    label_save_csv = osp.join('ROBINv1.1-cls-pose', folder, 'labels.csv')
    labels = {
        'imgs': [],
        'labels': [],
        'azimuth': [],
        'elevation': [],
        'theta': [],
        'distance': [],
    }
    pbar = trange(len(df), )
    pbar.set_description(f'Processing {folder}')
    for idx, row in df.iterrows():
        img = Image.open(osp.join('ROBINv1.1/', folder, row.im_path))
        img = img.crop((row.left - 10, row.upper - 10, row.right + 10, row.lower + 10))
        img_name = '_'.join(list(map(str, [row.source, row.cls_name, row.im_name, row.object, ]))) + '.jpg'
        img.convert('RGB').save(osp.join('ROBINv1.1-cls-pose/', folder, 'Images', row.cls_name, img_name))
        labels['imgs'].append(img_name)
        labels['labels'].append(row.cls_name)
        labels['azimuth'].append(row.azimuth)
        labels['elevation'].append(row.elevation)
        labels['theta'].append(row.inplane_rotation)
        labels['distance'].append(row.distance)
        
        pbar.update()
        
    labels_df = pd.DataFrame(labels)
    labels_df.to_csv(label_save_csv, index=None)
    
# create object detection dataset
os.makedirs('ROBINv1.1-det', exist_ok=True)

for folder in file_folders:
    df = pd.read_csv(glob(osp.join('ROBINv1.1/', folder, '*.csv'))[0])
    img_save_dir = osp.join('ROBINv1.1-det/', folder, 'Images')
    anno_save_dir = osp.join('ROBINv1.1-det/', folder, 'Annotations')
    os.makedirs(img_save_dir, exist_ok=True)
    os.makedirs(anno_save_dir, exist_ok=True)
    
    img_names = list(set(df.im_name))
    pbar = trange(len(df), )
    pbar.set_description(f'Processing {folder}')
    for img_name in img_names:
        sub_df = df[df.im_name == img_name]
        
        # this loop to get the im_path
        for idx, row in sub_df.iterrows():
            img_path = osp.join('ROBINv1.1/', folder, row.im_path)
            img = Image.open(img_path).convert('RGB')
            break
        writer = Writer(img_path, img.width, img.height)
        for idx, row in sub_df.iterrows():
            writer.addObject(row.cls_name, row.left, row.upper, row.right, row.lower)
            
        img.save(osp.join('ROBINv1.1-det/', folder, 'Images', img_name + '.jpg'))
        writer.save(osp.join('ROBINv1.1-det/', folder, 'Annotations', img_name + '.xml'))
        pbar.update()
