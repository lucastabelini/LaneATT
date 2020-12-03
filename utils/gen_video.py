import pickle
import argparse

import cv2
import numpy as np
from tqdm import tqdm

from lib.config import Config


def parse_args():
    parser = argparse.ArgumentParser(description="Tool to generate qualitative results videos")
    parser.add_argument("--pred", help=".pkl file to load predictions from", required=True)
    parser.add_argument("--cfg", default="config.yaml", help="Config file")
    parser.add_argument("--cover", default="tusimple_cover.png", help="Cover image file")
    parser.add_argument("--out", default="video.avi", help="Output filename")
    parser.add_argument("--view", action="store_true", help="Show predictions instead of creating video")
    parser.add_argument("--length", type=int, help="Length of the output video (seconds)")
    parser.add_argument("--clips", type=int, help="Number of clips")
    parser.add_argument("--fps", default=5, type=int, help="Video FPS")
    parser.add_argument("--legend", help="Path to legend image file")

    return parser.parse_args()


def add_cover_img(video, cover_path, frames=90):
    cover = cv2.imread(cover_path)
    for _ in range(frames):
        video.write(cover)


def create_video(filename, width, height, fps=5):
    fourcc = cv2.VideoWriter_fourcc(*'MP42')
    video = cv2.VideoWriter(filename, fourcc, float(fps), (width, height))

    return video


def main():
    np.random.seed(0)
    args = parse_args()
    cfg = Config(args.cfg)
    print('Loading dataset...')
    dataset = cfg.get_dataset('test')
    print('Done')
    height, width = cfg['datasets']['test']['parameters']['img_size']
    print('Using resolution {}x{}'.format(width, height))
    legend = cv2.imread(args.legend) if args.legend else None
    if not args.view:
        video = create_video(args.out, width, height + legend.shape[0] if legend is not None else 0, args.fps)

    print('Loading predictions...')
    with open(args.pred, "rb") as pred_file:
        predictions = np.array(pickle.load(pred_file))
    print("Done.")

    if args.length is not None and args.clips is not None:
        video_length = args.length * args.fps
        assert video_length % args.clips == 0
        clip_length = video_length // args.clips
        all_clip_ids = np.arange(len(dataset) // clip_length)
        selected_clip_ids = np.random.choice(all_clip_ids, size=args.clips, replace=False)
        frame_idxs = (np.repeat(selected_clip_ids, clip_length).reshape(args.clips, clip_length) + np.arange(
            clip_length)).flatten()
        total = len(frame_idxs)
    else:
        total = len(dataset)
        frame_idxs = np.arange(len(dataset))

    for idx, pred in tqdm(zip(frame_idxs, predictions[frame_idxs]), total=total):
        frame, _, _ = dataset.draw_annotation(idx, pred=pred)
        assert frame.shape[:2] == (height, width)
        if legend is not None:
            frame = np.vstack((legend, frame))
        if args.view:
            cv2.imshow('frame', frame)
            cv2.waitKey(0)
        else:
            video.write(frame)

    if not args.view:
        video.release()
        print('Video saved as {}'.format(args.out))


if __name__ == '__main__':
    main()
