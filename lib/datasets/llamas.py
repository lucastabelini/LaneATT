import os
import pickle as pkl

import numpy as np
from tqdm import tqdm

from .lane_dataset_loader import LaneDatasetLoader

TRAIN_LABELS_DIR = 'labels/train'
TEST_LABELS_DIR = 'labels/valid'
TEST_IMGS_DIR = 'color_images/test'
SPLIT_DIRECTORIES = {'train': 'labels/train', 'val': 'labels/valid'}
from utils.llamas_utils import get_horizontal_values_for_four_lanes
import utils.llamas_metric as llamas_metric


class LLAMAS(LaneDatasetLoader):
    def __init__(self, split='train', max_lanes=None, root=None):
        self.split = split
        self.root = root
        if split != 'test' and split not in SPLIT_DIRECTORIES.keys():
            raise Exception('Split `{}` does not exist.'.format(split))
        if split != 'test':
            self.labels_dir = os.path.join(self.root, SPLIT_DIRECTORIES[split])

        self.img_w, self.img_h = 1276, 717
        self.annotations = []
        self.load_annotations()

        # Force max_lanes, used when evaluating testing with models trained on other datasets
        if max_lanes is not None:
            self.max_lanes = max_lanes

    def get_img_heigth(self, _):
        return self.img_h

    def get_img_width(self, _):
        return self.img_w

    def get_metrics(self, lanes, _):
        # Placeholders
        return [0] * len(lanes), [0] * len(lanes), [1] * len(lanes), [1] * len(lanes)

    def get_img_path(self, json_path):
        # /foo/bar/test/folder/image_label.ext --> test/folder/image_label.ext
        base_name = '/'.join(json_path.split('/')[-3:])
        image_path = os.path.join('color_images', base_name.replace('.json', '_color_rect.png'))
        return image_path

    def get_json_paths(self):
        json_paths = []
        for root, _, files in os.walk(self.labels_dir):
            for file in files:
                if file.endswith(".json"):
                    json_paths.append(os.path.join(root, file))
        return json_paths

    def load_annotations(self):
        # the labels are not public for the test set yet
        if self.split == 'test':
            imgs_dir = os.path.join(self.root, TEST_IMGS_DIR)
            self.annotations = [{
                'path': os.path.join(root, file),
                'lanes': [],
                'relative_path': file
            } for root, _, files in os.walk(imgs_dir) for file in files if file.endswith('.png')]
            self.annotations = sorted(self.annotations, key=lambda x: x['path'])
            return
        # Waiting for the dataset to load is tedious, let's cache it
        os.makedirs('cache', exist_ok=True)
        cache_path = 'cache/llamas_{}.pkl'.format(self.split)
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as cache_file:
                self.annotations = pkl.load(cache_file)
                self.max_lanes = max(len(anno['lanes']) for anno in self.annotations)
                return

        self.max_lanes = 0
        print("Searching annotation files...")
        json_paths = self.get_json_paths()
        print('{} annotations found.'.format(len(json_paths)))

        for json_path in tqdm(json_paths):
            lanes = get_horizontal_values_for_four_lanes(json_path)
            lanes = [[(x, y) for x, y in zip(lane, range(self.img_h)) if x >= 0] for lane in lanes]
            lanes = [lane for lane in lanes if len(lane) > 0]
            relative_path = self.get_img_path(json_path)
            img_path = os.path.join(self.root, relative_path)
            self.max_lanes = max(self.max_lanes, len(lanes))
            self.annotations.append({'path': img_path, 'lanes': lanes, 'aug': False, 'relative_path': relative_path})

        with open(cache_path, 'wb') as cache_file:
            pkl.dump(self.annotations, cache_file)

    def assign_class_to_lanes(self, lanes):
        return {label: value for label, value in zip(['l0', 'l1', 'r0', 'r1'], lanes)}

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
            relative_path = self.annotations[idx]['old_anno']['relative_path']
            output_filename = '/'.join(relative_path.split('/')[-2:]).replace('_color_rect.png', '.lines.txt')
            output_filepath = os.path.join(output_basedir, output_filename)
            os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
            output = self.get_prediction_string(pred)
            with open(output_filepath, 'w') as out_file:
                out_file.write(output)
        if self.split == 'test':
            return {}
        return llamas_metric.eval_predictions(output_basedir, self.labels_dir, unofficial=False)

    def __getitem__(self, idx):
        return self.annotations[idx]

    def __len__(self):
        return len(self.annotations)
