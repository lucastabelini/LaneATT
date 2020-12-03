import os
import json
import logging

import numpy as np
from tqdm import tqdm

import utils.culane_metric as culane_metric

from .lane_dataset_loader import LaneDatasetLoader

SPLIT_FILES = {
    'train': "list/train.txt",
    'val': 'list/val.txt',
    'test': "list/test.txt",
    'normal': 'list/test_split/test0_normal.txt',
    'crowd': 'list/test_split/test1_crowd.txt',
    'hlight': 'list/test_split/test2_hlight.txt',
    'shadow': 'list/test_split/test3_shadow.txt',
    'noline': 'list/test_split/test4_noline.txt',
    'arrow': 'list/test_split/test5_arrow.txt',
    'curve': 'list/test_split/test6_curve.txt',
    'cross': 'list/test_split/test7_cross.txt',
    'night': 'list/test_split/test8_night.txt',
    'debug': 'list/debug.txt'
}


class CULane(LaneDatasetLoader):
    def __init__(self, max_lanes=None, split='train', root=None, official_metric=True):
        self.split = split
        self.root = root
        self.official_metric = official_metric
        self.logger = logging.getLogger(__name__)

        if root is None:
            raise Exception('Please specify the root directory')
        if split not in SPLIT_FILES:
            raise Exception('Split `{}` does not exist.'.format(split))

        self.list = os.path.join(root, SPLIT_FILES[split])

        self.img_w, self.img_h = 1640, 590
        self.annotations = []
        self.load_annotations()
        self.max_lanes = 4 if max_lanes is None else max_lanes

    def get_img_heigth(self, _):
        return self.img_h

    def get_img_width(self, _):
        return self.img_w

    def get_metrics(self, raw_lanes, idx):
        lanes = []
        pred_str = self.get_prediction_string(raw_lanes)
        for lane in pred_str.split('\n'):
            if lane == '':
                continue
            lane = list(map(float, lane.split()))
            lane = [(lane[i], lane[i + 1]) for i in range(0, len(lane), 2) if lane[i] >= 0 and lane[i + 1] >= 0]
            lanes.append(lane)
        anno = culane_metric.load_culane_img_data(self.annotations[idx]['path'].replace('.jpg', '.lines.txt'))
        _, fp, fn, ious, matches = culane_metric.culane_metric(lanes, anno)

        return fp, fn, matches, ious

    def load_annotation(self, img_path):
        anno_path = img_path[:-3] + 'lines.txt'  # remove sufix jpg and add lines.txt

        with open(anno_path, 'r') as anno_file:
            data = [list(map(float, line.split())) for line in anno_file.readlines()]

        lanes = [[(lane[i], lane[i + 1]) for i in range(0, len(lane), 2) if lane[i] >= 0 and lane[i + 1] >= 0]
                 for lane in data]
        lanes = [list(set(lane)) for lane in lanes]  # remove duplicated points
        lanes = [lane for lane in lanes if len(lane) >= 2]  # remove lanes with less than 2 points

        lanes = [sorted(lane, key=lambda x: x[1]) for lane in lanes]  # sort by y

        return {'path': img_path, 'lanes': lanes}

    def load_annotations(self):
        self.annotations = []
        self.max_lanes = 0
        os.makedirs('cache', exist_ok=True)
        cache_path = 'cache/culane_{}.json'.format(self.split)

        if os.path.exists(cache_path):
            self.logger.info('Loading CULane annotations (cached)...')
            with open(cache_path, 'r') as cache_file:
                data = json.load(cache_file)
                self.annotations = data['annotations']
                self.max_lanes = data['max_lanes']
        else:
            self.logger.info('Loading CULane annotations and caching...')
            with open(self.list, 'r') as list_file:
                files = [line.rstrip()[1 if line[0] == '/' else 0::]
                         for line in list_file]  # remove `/` from beginning if needed

            for file in tqdm(files):
                img_path = os.path.join(self.root, file)
                anno = self.load_annotation(img_path)
                anno['org_path'] = file

                if len(anno['lanes']) > 0:
                    self.max_lanes = max(self.max_lanes, len(anno['lanes']))
                self.annotations.append(anno)
            with open(cache_path, 'w') as cache_file:
                json.dump({'annotations': self.annotations, 'max_lanes': self.max_lanes}, cache_file)

        self.logger.info('%d annotations loaded, with a maximum of %d lanes in an image.', len(self.annotations),
                         self.max_lanes)

    def get_prediction_string(self, pred):
        ys = np.arange(self.img_h) / self.img_h
        out = []
        for lane in pred:
            xs = lane(ys)
            valid_mask = (xs >= 0) & (xs < 1)
            xs = xs * self.img_w
            lane_xs = xs[valid_mask]
            lane_ys = ys[valid_mask] * self.img_h
            lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]
            lane_str = ' '.join(['{:.5f} {:.5f}'.format(x, y) for x, y in zip(lane_xs, lane_ys)])
            if lane_str != '':
                out.append(lane_str)

        return '\n'.join(out)

    def eval_predictions(self, predictions, output_basedir):
        print('Generating prediction output...')
        for idx, pred in enumerate(tqdm(predictions)):
            output_dir = os.path.join(output_basedir, os.path.dirname(self.annotations[idx]['old_anno']['org_path']))
            output_filename = os.path.basename(self.annotations[idx]['old_anno']['org_path'])[:-3] + 'lines.txt'
            os.makedirs(output_dir, exist_ok=True)
            output = self.get_prediction_string(pred)
            with open(os.path.join(output_dir, output_filename), 'w') as out_file:
                out_file.write(output)
        return culane_metric.eval_predictions(output_basedir, self.root, self.list, official=self.official_metric)

    def transform_annotations(self, transform):
        self.annotations = list(map(transform, self.annotations))

    def __getitem__(self, idx):
        return self.annotations[idx]

    def __len__(self):
        return len(self.annotations)
