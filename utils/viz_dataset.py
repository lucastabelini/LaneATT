import argparse

import cv2
import torch
import random
import numpy as np

from lib.config import Config


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize a dataset")
    parser.add_argument("--cfg", help="Config file")
    parser.add_argument("--split",
                        choices=["train", "test", "val"],
                        default='train',
                        help="Dataset split to visualize")
    args = parser.parse_args()

    return args


def main():
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
    args = parse_args()
    cfg = Config(args.cfg)
    train_dataset = cfg.get_dataset(args.split)
    for idx in range(len(train_dataset)):
        img, _, _ = train_dataset.draw_annotation(idx)
        cv2.imshow('sample', img)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
