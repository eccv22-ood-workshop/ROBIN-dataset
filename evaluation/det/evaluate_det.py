import argparse
import json
import os
import os.path as osp

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to the output dir')
parser.add_argument('--input', default='', type=str, metavar='PATH',
                    help='path to the input dir')
parser.add_argument('--iid-perf', default=1.0, type=float,
                    help='iid performance threshold')


def eval_one_nuisance(label_json_path, result_json_path):
    coco_gt = COCO(label_json_path)
    coco_dt = coco_gt.loadRes(result_json_path)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats[0]  # AP50


def main():
    args = parser.parse_args()
    print(args)

    label_path = os.path.join(args.input, "ref")
    result_path = os.path.join(args.input, "res")
    print("Reading result from ", result_path)

    print("Evaluating IID")
    iid_ap50 = eval_one_nuisance(
        osp.join(label_path, "iid_test.json"),
        osp.join(result_path, "iid_test.json")
    )
    print("Current iid performance: ", iid_ap50)
    assert iid_ap50 <= args.iid_perf, f"Excceed IID accuracy threshold {args.iid_perf}"

    nuisances = ['shape', 'pose', 'texture', 'context', 'weather', 'occlusion']
    ap50s = {}

    for nuisance in nuisances:
        print(f"Evaluating {nuisance}")
        ap50s[nuisance] = eval_one_nuisance(
            osp.join(label_path, f"{nuisance}.json"),
            osp.join(result_path, f"{nuisance}.json")
        )
        print(f"ap50@{nuisance}: {ap50s[nuisance]}")

    mean_ap50 = sum(ap50s.values()) / len(ap50s)
    print("Mean-AP50: ", mean_ap50)

    output_path = os.path.join(args.output, "scores.txt")
    print("Writing scores to ", output_path)
    with open(output_path, mode="w") as f:
        for nuisance in nuisances:
            print(f'OOD-{nuisance}-AP50: ', ap50s[nuisance], file=f)
        print("OOD-AP50: ", mean_ap50, file=f)
        print("IID-AP50: ", iid_ap50, file=f)


if __name__ == '__main__':
    main()