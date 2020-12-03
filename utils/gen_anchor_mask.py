import random
import argparse

import cv2
import torch
import numpy as np
from tqdm import trange

from lib.config import Config
from lib.models.matching import match_proposals_with_targets


def get_anchors_use_frequency(cfg, split='train', t_pos=15., t_neg=20.):
    model = cfg.get_model()
    anchors_frequency = torch.zeros(len(model.anchors), dtype=torch.int32)
    nb_unmatched_targets = 0
    dataset = cfg.get_dataset(split)
    for idx in trange(len(dataset)):
        _, targets, _ = dataset[idx]
        targets = targets[targets[:, 1] == 1]
        n_targets = len(targets)
        if n_targets == 0:
            continue
        targets = torch.tensor(targets)
        positives_mask, _, _, target_indices = match_proposals_with_targets(model,
                                                                            model.anchors,
                                                                            targets,
                                                                            t_pos=t_pos,
                                                                            t_neg=t_neg)
        n_matches = len(set(target_indices.tolist()))
        nb_unmatched_targets += n_targets - n_matches
        assert (n_targets - n_matches) >= 0

        anchors_frequency += positives_mask

    return anchors_frequency


def save_mask(cfg_path, output_path):
    cfg = Config(cfg_path)
    frequency = get_anchors_use_frequency(cfg, split='train', t_pos=30., t_neg=35.)
    torch.save(frequency, output_path)


def view_mask():
    cfg = Config('config.yaml')
    model = cfg.get_model()
    img = model.draw_anchors(img_w=512, img_h=288)
    cv2.imshow('anchors', img)
    cv2.waitKey(0)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute anchor frequency for later use as anchor mask")
    parser.add_argument("--output", help="Output path (e.g., `anchors_mask.pt`", required=True)
    parser.add_argument("--cfg", help="Config file (e.g., `config.yml`")
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    # Fix seeds
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    save_mask(args.cfg, args.output)
    # view_mask()
