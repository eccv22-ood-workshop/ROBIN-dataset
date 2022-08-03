import argparse
import json
import pandas as pd
import os
import os.path as osp


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to the output dir')
parser.add_argument('--input', default='', type=str, metavar='PATH',
                    help='path to the input dir')
parser.add_argument('--iid-perf', default=0.911, type=float,
                    help='iid performance threshold')


def main():
    args = parser.parse_args()
    print(args)
    
    iid_gt = pd.read_csv(osp.join(args.input, 'ref', 'iid_test', 'labels.csv'))
    iid_pred = pd.read_csv(osp.join(args.input, 'res', 'iid.csv'))

    iid_acc = (iid_gt['labels'] == iid_pred['pred']).sum() / len(iid_gt['labels'])
    print("Current iid performance: ", iid_acc)
    assert iid_acc <= args.iid_perf, f"Excceed IID accuracy threshold {args.iid_perf}"

    nuisances = ['shape', 'pose', 'texture', 'context', 'weather', 'occlusion']
    accs = {}

    for nuisance in nuisances:
        gt = pd.read_csv(osp.join(args.input, 'ref', 'nuisances', nuisance, 'labels.csv'))
        pred = pd.read_csv(osp.join(args.input, 'res', nuisance + '.csv'))
        accs[nuisance] = (gt['labels'] == pred['pred']).sum() / len(gt['labels'])
        print(f"TOP-1@{nuisance}: {accs[nuisance]}")

    mean_acc = sum(accs.values()) / len(accs)
    print("Mean-TOP-1: ", mean_acc)

    output_path = os.path.join(args.output, "scores.txt")
    print("Writing scores to ", output_path)
    with open(output_path, mode="w") as f:
        for nuisance in nuisances:
            print(f'OOD-{nuisance}-TOP-1: ', accs[nuisance], file=f)
        print("OOD-TOP-1: ", mean_acc, file=f)
        print("IID-TOP-1: ", iid_acc, file=f)



if __name__ == '__main__':
    main()

    # import time
    # time.sleep(3 * 60)