import os
import argparse
from functools import partial

import cv2
import numpy as np
from tqdm import tqdm
from p_tqdm import t_map, p_map
from scipy.interpolate import splprep, splev
from scipy.optimize import linear_sum_assignment
from shapely.geometry import LineString, Polygon


def draw_lane(lane, img=None, img_shape=None, width=30):
    if img is None:
        img = np.zeros(img_shape, dtype=np.uint8)
    lane = lane.astype(np.int32)
    for p1, p2 in zip(lane[:-1], lane[1:]):
        cv2.line(img, tuple(p1), tuple(p2), color=(255, 255, 255), thickness=width)
    return img


def discrete_cross_iou(xs, ys, width=30, img_shape=(590, 1640, 3)):
    xs = [draw_lane(lane, img_shape=img_shape, width=width) > 0 for lane in xs]
    ys = [draw_lane(lane, img_shape=img_shape, width=width) > 0 for lane in ys]

    ious = np.zeros((len(xs), len(ys)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            ious[i, j] = (x & y).sum() / (x | y).sum()
    return ious


def continuous_cross_iou(xs, ys, width=30, img_shape=(590, 1640, 3)):
    h, w, _ = img_shape
    image = Polygon([(0, 0), (0, h - 1), (w - 1, h - 1), (w - 1, 0)])
    xs = [LineString(lane).buffer(distance=width / 2., cap_style=1, join_style=2).intersection(image) for lane in xs]
    ys = [LineString(lane).buffer(distance=width / 2., cap_style=1, join_style=2).intersection(image) for lane in ys]

    ious = np.zeros((len(xs), len(ys)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            ious[i, j] = x.intersection(y).area / x.union(y).area

    return ious


def interp(points, n=50):
    x = [x for x, _ in points]
    y = [y for _, y in points]
    tck, u = splprep([x, y], s=0, t=n, k=min(3, len(points) - 1))

    u = np.linspace(0., 1., num=(len(u) - 1) * n + 1)
    return np.array(splev(u, tck)).T


def culane_metric(pred, anno, width=30, iou_threshold=0.5, official=True, img_shape=(590, 1640, 3)):
    if len(pred) == 0:
        return 0, 0, len(anno), np.zeros(len(pred)), np.zeros(len(pred), dtype=bool)
    if len(anno) == 0:
        return 0, len(pred), 0, np.zeros(len(pred)), np.zeros(len(pred), dtype=bool)
    interp_pred = np.array([interp(pred_lane, n=5) for pred_lane in pred], dtype=object)  # (4, 50, 2)
    interp_anno = np.array([interp(anno_lane, n=5) for anno_lane in anno], dtype=object)  # (4, 50, 2)

    if official:
        ious = discrete_cross_iou(interp_pred, interp_anno, width=width, img_shape=img_shape)
    else:
        ious = continuous_cross_iou(interp_pred, interp_anno, width=width, img_shape=img_shape)

    row_ind, col_ind = linear_sum_assignment(1 - ious)
    tp = int((ious[row_ind, col_ind] > iou_threshold).sum())
    fp = len(pred) - tp
    fn = len(anno) - tp
    pred_ious = np.zeros(len(pred))
    pred_ious[row_ind] = ious[row_ind, col_ind]
    return tp, fp, fn, pred_ious, pred_ious > iou_threshold


def load_culane_img_data(path):
    with open(path, 'r') as data_file:
        img_data = data_file.readlines()
    img_data = [line.split() for line in img_data]
    img_data = [list(map(float, lane)) for lane in img_data]
    img_data = [[(lane[i], lane[i + 1]) for i in range(0, len(lane), 2)] for lane in img_data]
    img_data = [lane for lane in img_data if len(lane) >= 2]

    return img_data


def load_culane_data(data_dir, file_list_path):
    with open(file_list_path, 'r') as file_list:
        filepaths = [
            os.path.join(data_dir, line[1 if line[0] == '/' else 0:].rstrip().replace('.jpg', '.lines.txt'))
            for line in file_list.readlines()
        ]

    data = []
    for path in tqdm(filepaths):
        img_data = load_culane_img_data(path)
        data.append(img_data)

    return data


def eval_predictions(pred_dir, anno_dir, list_path, width=30, official=True, sequential=False):
    print('List: {}'.format(list_path))
    print('Loading prediction data...')
    predictions = load_culane_data(pred_dir, list_path)
    print('Loading annotation data...')
    annotations = load_culane_data(anno_dir, list_path)
    print('Calculating metric {}...'.format('sequentially' if sequential else 'in parallel'))
    img_shape = (590, 1640, 3)
    if sequential:
        results = t_map(partial(culane_metric, width=width, official=official, img_shape=img_shape), predictions,
                        annotations)
    else:
        results = p_map(partial(culane_metric, width=width, official=official, img_shape=img_shape), predictions,
                        annotations)
    total_tp = sum(tp for tp, _, _, _, _ in results)
    total_fp = sum(fp for _, fp, _, _, _ in results)
    total_fn = sum(fn for _, _, fn, _, _ in results)
    if total_tp == 0:
        precision = 0
        recall = 0
        f1 = 0
    else:
        precision = float(total_tp) / (total_tp + total_fp)
        recall = float(total_tp) / (total_tp + total_fn)
        f1 = 2 * precision * recall / (precision + recall)

    return {'TP': total_tp, 'FP': total_fp, 'FN': total_fn, 'Precision': precision, 'Recall': recall, 'F1': f1}


def main():
    args = parse_args()
    for list_path in args.list:
        results = eval_predictions(args.pred_dir,
                                   args.anno_dir,
                                   list_path,
                                   width=args.width,
                                   official=args.official,
                                   sequential=args.sequential)

        header = '=' * 20 + ' Results ({})'.format(os.path.basename(list_path)) + '=' * 20
        print(header)
        for metric, value in results.items():
            if isinstance(value, float):
                print('{}: {:.4f}'.format(metric, value))
            else:
                print('{}: {}'.format(metric, value))
        print('=' * len(header))


def parse_args():
    parser = argparse.ArgumentParser(description="Measure CULane's metric")
    parser.add_argument("--pred_dir", help="Path to directory containing the predicted lanes", required=True)
    parser.add_argument("--anno_dir", help="Path to directory containing the annotated lanes", required=True)
    parser.add_argument("--width", type=int, default=30, help="Width of the lane")
    parser.add_argument("--list", nargs='+', help="Path to txt file containing the list of files", required=True)
    parser.add_argument("--sequential", action='store_true', help="Run sequentially instead of in parallel")
    parser.add_argument("--official", action='store_true', help="Use official way to calculate the metric")

    return parser.parse_args()


if __name__ == '__main__':
    main()
