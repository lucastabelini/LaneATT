import os
import json
import random
import logging

import numpy as np

from utils.tusimple_metric import LaneEval

from .lane_dataset_loader import LaneDatasetLoader

SPLIT_FILES = {
    'train+val': ['label_data_0313.json', 'label_data_0601.json', 'label_data_0531.json'],
    'train': ['label_data_0313.json', 'label_data_0601.json'],
    'val': ['label_data_0531.json'],
    'test': ['test_label.json'],
}


class TuSimple(LaneDatasetLoader):
    def __init__(self, split='train', max_lanes=None, root=None):
        self.split = split
        self.root = root
        self.logger = logging.getLogger(__name__)

        if split not in SPLIT_FILES.keys():
            raise Exception('Split `{}` does not exist.'.format(split))

        self.anno_files = [os.path.join(self.root, path) for path in SPLIT_FILES[split]]

        if root is None:
            raise Exception('Please specify the root directory')

        self.img_w, self.img_h = 1280, 720
        self.annotations = []
        self.load_annotations()

        # Force max_lanes, used when evaluating testing with models trained on other datasets
        if max_lanes is not None:
            self.max_lanes = max_lanes

    def get_img_heigth(self, _):
        return 720

    def get_img_width(self, _):
        return 1280

    def get_metrics(self, lanes, idx):
        label = self.annotations[idx]
        org_anno = label['old_anno']
        pred = self.pred2lanes(org_anno['path'], lanes, org_anno['y_samples'])
        _, fp, fn, matches, accs, _ = LaneEval.bench(pred, org_anno['org_lanes'], org_anno['y_samples'], 0, True)
        return fp, fn, matches, accs

    def pred2lanes(self, path, pred, y_samples):
        ys = np.array(y_samples) / self.img_h
        lanes = []
        for lane in pred:
            xs = lane(ys)
            invalid_mask = xs < 0
            lane = (xs * self.get_img_width(path)).astype(int)
            lane[invalid_mask] = -2
            lanes.append(lane.tolist())

        return lanes

    def load_annotations(self):
        self.logger.info('Loading TuSimple annotations...')
        self.annotations = []
        max_lanes = 0
        for anno_file in self.anno_files:
            with open(anno_file, 'r') as anno_obj:
                lines = anno_obj.readlines()
            for line in lines:
                data = json.loads(line)
                y_samples = data['h_samples']
                gt_lanes = data['lanes']
                lanes = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in gt_lanes]
                lanes = [lane for lane in lanes if len(lane) > 0]
                max_lanes = max(max_lanes, len(lanes))
                self.annotations.append({
                    'path': os.path.join(self.root, data['raw_file']),
                    'org_path': data['raw_file'],
                    'org_lanes': gt_lanes,
                    'lanes': lanes,
                    'aug': False,
                    'y_samples': y_samples
                })

        if self.split == 'train':
            random.shuffle(self.annotations)
        self.max_lanes = max_lanes
        self.logger.info('%d annotations loaded, with a maximum of %d lanes in an image.', len(self.annotations),
                         self.max_lanes)

    def transform_annotations(self, transform):
        self.annotations = list(map(transform, self.annotations))

    def pred2tusimpleformat(self, idx, pred, runtime):
        runtime *= 1000.  # s to ms
        img_name = self.annotations[idx]['old_anno']['org_path']
        h_samples = self.annotations[idx]['old_anno']['y_samples']
        lanes = self.pred2lanes(img_name, pred, h_samples)
        output = {'raw_file': img_name, 'lanes': lanes, 'run_time': runtime}
        return json.dumps(output)

    def save_tusimple_predictions(self, predictions, filename, runtimes=None):
        if runtimes is None:
            runtimes = np.ones(len(predictions)) * 1.e-3
        lines = []
        for idx, (prediction, runtime) in enumerate(zip(predictions, runtimes)):
            line = self.pred2tusimpleformat(idx, prediction, runtime)
            lines.append(line)
        with open(filename, 'w') as output_file:
            output_file.write('\n'.join(lines))

    def eval_predictions(self, predictions, output_basedir, runtimes=None):
        pred_filename = os.path.join(output_basedir, 'tusimple_predictions.json')
        self.save_tusimple_predictions(predictions, pred_filename, runtimes)
        result = json.loads(LaneEval.bench_one_submit(pred_filename, self.anno_files[0]))
        table = {}
        for metric in result:
            table[metric['name']] = metric['value']

        return table

    def __getitem__(self, idx):
        return self.annotations[idx]

    def __len__(self):
        return len(self.annotations)
