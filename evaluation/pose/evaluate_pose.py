import argparse
import json
import pandas as pd
import os
import os.path as osp
import numpy as np
import math
from scipy.linalg import logm

def get_transformation_matrix(azimuth, elevation, distance):
    if distance == 0:
        # return None
        distance = 0.1

    # camera center
    C = np.zeros((3, 1))
    C[0] = distance * math.cos(elevation) * math.sin(azimuth)
    C[1] = -distance * math.cos(elevation) * math.cos(azimuth)
    C[2] = distance * math.sin(elevation)

    # rotate coordinate system by theta is equal to rotating the model by theta
    azimuth = -azimuth
    elevation = - (math.pi / 2 - elevation)

    # rotation matrix
    Rz = np.array([
        [math.cos(azimuth), -math.sin(azimuth), 0],
        [math.sin(azimuth), math.cos(azimuth), 0],
        [0, 0, 1],
    ])  # rotation by azimuth
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(elevation), -math.sin(elevation)],
        [0, math.sin(elevation), math.cos(elevation)],
    ])  # rotation by elevation
    R_rot = np.dot(Rx, Rz)
    R = np.hstack((R_rot, np.dot(-R_rot, C)))
    R = np.vstack((R, [0, 0, 0, 1]))

    return R

def rotation_theta(theta):
    # cos -sin  0
    # sin  cos  0
    # 0    0    1
    return np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])


def cal_err(gt, pred):
    # return radius
    return ((logm(np.dot(np.transpose(pred), gt)) ** 2).sum()) ** 0.5 / (2. ** 0.5)


def cal_rotation_matrix(theta, elev, azum, dis):
    if dis <= 1e-10:
        dis = 0.5

    return rotation_theta(theta) @ get_transformation_matrix(azum, elev, dis)[0:3, 0:3]




parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to the output dir')
parser.add_argument('--input', default='', type=str, metavar='PATH',
                    help='path to the input dir')
parser.add_argument('--iid-perf', default=0.687, type=float,
                    help='iid performance threshold')


def main():
    args = parser.parse_args()
    print(args)
    
    iid_gt = pd.read_csv(osp.join(args.input, 'ref', 'iid_test', 'labels.csv'))
    iid_pred = pd.read_csv(osp.join(args.input, 'res', 'iid.csv'))
    thr = np.pi / 6
    

    iid_acc = 0

    theta_anno, elevation_anno, azimuth_anno, distance_anno = [], [], [], []
    for idx, row in iid_gt.iterrows():
        theta_anno.append(row.theta)
        elevation_anno.append(row.elevation)
        azimuth_anno.append(row.azimuth)
        distance_anno.append(row.distance)

    theta_pred, elevation_pred, azimuth_pred, distance_pred = [], [], [], []
    for idx, row in iid_pred.iterrows():
        theta_pred.append(row.theta)
        elevation_pred.append(row.elevation)
        azimuth_pred.append(row.azimuth)
        distance_pred.append(row.distance)


    iid_error = []
    for theta_p, theta_a, elevation_p, elevation_a, azimuth_p, azimuth_a, distance_p, distance_a in zip(theta_pred, theta_anno, 
                                                                                                        elevation_pred, elevation_anno, 
                                                                                                        azimuth_pred, azimuth_anno, 
                                                                                                        distance_pred, distance_anno):
        anno_matrix = cal_rotation_matrix(theta_a, elevation_a, azimuth_a, distance_a)
        pred_matrix = cal_rotation_matrix(theta_p, elevation_p, azimuth_p, distance_p)
        if np.any(np.isnan(anno_matrix)) or np.any(np.isnan(pred_matrix)) or np.any(np.isinf(anno_matrix)) or np.any(np.isinf(pred_matrix)):
            error_ = np.pi / 2
        else:
            error_ = cal_err(anno_matrix, pred_matrix)
        iid_error.append(error_)
    iid_error = np.array(iid_error)
    
    iid_acc = float(np.mean(iid_error < thr)) 
        

    print("Current iid performance: ", iid_acc)
    assert iid_acc <= args.iid_perf, f"Excceed IID accuracy threshold {args.iid_perf}"

    nuisances = ['shape', 'pose', 'texture', 'context', 'weather', 'occlusion']
    accs = {}

    for nuisance in nuisances:
        gt = pd.read_csv(osp.join(args.input, 'ref', 'nuisances', nuisance, 'labels.csv'))
        pred = pd.read_csv(osp.join(args.input, 'res', nuisance + '.csv'))

        theta_anno, elevation_anno, azimuth_anno, distance_anno = [], [], [], []
        for idx, row in gt.iterrows():
            theta_anno.append(row.theta)
            elevation_anno.append(row.elevation)
            azimuth_anno.append(row.azimuth)
            distance_anno.append(row.distance)

        theta_pred, elevation_pred, azimuth_pred, distance_pred = [], [], [], []
        for idx, row in pred.iterrows():
            theta_pred.append(row.theta)
            elevation_pred.append(row.elevation)
            azimuth_pred.append(row.azimuth)
            distance_pred.append(row.distance)

        
        error = []
        for theta_p, theta_a, elevation_p, elevation_a, azimuth_p, azimuth_a, distance_p, distance_a in zip(theta_pred, theta_anno, 
                                                                                                            elevation_pred, elevation_anno, 
                                                                                                            azimuth_pred, azimuth_anno, 
                                                                                                            distance_pred, distance_anno):
            anno_matrix = cal_rotation_matrix(theta_a, elevation_a, azimuth_a, distance_a)
            pred_matrix = cal_rotation_matrix(theta_p, elevation_p, azimuth_p, distance_p)
            if np.any(np.isnan(anno_matrix)) or np.any(np.isnan(pred_matrix)) or np.any(np.isinf(anno_matrix)) or np.any(np.isinf(pred_matrix)):
                error_ = np.pi / 2
            else:
                error_ = cal_err(anno_matrix, pred_matrix)
            error.append(error_)
        error = np.array(error)

        accs[nuisance] = float(np.mean(iid_error < thr)) 
        print(f"Acc@pi/6@{nuisance}: {accs[nuisance]}")

    mean_acc = sum(accs.values()) / len(accs)
    print("Mean-Acc@pi/6: ", mean_acc)

    output_path = os.path.join(args.output, "scores.txt")
    print("Writing scores to ", output_path)
    with open(output_path, mode="w") as f:
        for nuisance in nuisances:
            print(f'OOD-{nuisance}-TOP-1: ', accs[nuisance], file=f)
        print("OOD-Acc@pi/6: ", mean_acc, file=f)
        print("IID-Acc@pi/6: ", iid_acc, file=f)



if __name__ == '__main__':
    main()

    # import time
    # time.sleep(3 * 60)