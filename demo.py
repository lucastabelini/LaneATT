import argparse
import numpy

import torch
import cv2
from lib.config import Config
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Train lane detector")
    parser.add_argument(
        "mode", choices=["video", "image"], help="Video or image?")
    parser.add_argument("-path", help="image/video path", required=True)
    parser.add_argument("-cfg", help="Config file", required=True)
    args = parser.parse_args()
    return args


def process(model, image, img_w, img_h, cfg, device):
    image = cv2.resize(image, (img_w, img_h))
    data = torch.from_numpy(image.astype(numpy.float32)
                            ).permute(2, 0, 1) / 255.
    data = data.reshape(1, 3, img_h, img_w)
    images = data.to(device)
    test_parameters = cfg.get_test_parameters()
    output = model(images, **test_parameters)
    prediction = model.decode(output, as_lanes=True)
    for i, l in enumerate(prediction[0]):
        color = (0, 0, 255)
        points = l.points
        points[:, 0] *= image.shape[1]
        points[:, 1] *= image.shape[0]
        points = points.round().astype(int)
        # points += pad
        xs, ys = points[:, 0], points[:, 1]
        for curr_p, next_p in zip(points[:-1], points[1:]):
            image = cv2.line(image,
                             tuple(curr_p),
                             tuple(next_p),
                             color=color,
                             thickness=3)
    cv2.imshow("", image)


def main():
    args = parse_args()

    cfg_path = args.cfg
    cfg = Config(cfg_path)
    device = torch.device(
        'cpu') if not torch.cuda.is_available()else torch.device('cuda')

    model = cfg.get_model()
    check_point = "./experiments/laneatt_"+"r"+cfg["model"]["parameters"]["backbone"][6:]+"_"+cfg["datasets"]["train"]["parameters"]["dataset"] +"/models/"
    
    check_point+=os.listdir(check_point)[0]
    print("load check point:",check_point)
    dict = torch.load(check_point)
    model.load_state_dict(dict["model"])
    model = model.to(device)
    model.eval()
    img_h = cfg["model"]["parameters"]["img_h"]
    img_w = cfg["model"]["parameters"]["img_w"]
    if args.mode == "image":
        image = cv2.imread(args.path)
        process(model, image, img_w, img_h, cfg, device)
        cv2.waitKey(0)
    elif args.mode == "video":
        video = cv2.VideoCapture(args.path)
        while(True):
            rval, frame = video.read()
            process(model, frame, img_w, img_h, cfg, device)
            if cv2.waitKey(1) == 27:
                break


if __name__ == '__main__':
    main()
