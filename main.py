import logging
import argparse

import torch

from lib.config import Config
from lib.runner import Runner
from lib.experiment import Experiment


def parse_args():
    parser = argparse.ArgumentParser(description="Train lane detector")
    parser.add_argument("mode", choices=["train", "test"], help="Train or test?")
    parser.add_argument("--exp_name", help="Experiment name", required=True)
    parser.add_argument("--cfg", help="Config file")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("--epoch", type=int, help="Epoch to test the model on")
    parser.add_argument("--cpu", action="store_true", help="(Unsupported) Use CPU instead of GPU")
    parser.add_argument("--save_predictions", action="store_true", help="Save predictions to pickle file")
    parser.add_argument("--view", choices=["all", "mistakes"], help="Show predictions")
    parser.add_argument("--deterministic",
                        action="store_true",
                        help="set cudnn.deterministic = True and cudnn.benchmark = False")
    args = parser.parse_args()
    if args.cfg is None and args.mode == "train":
        raise Exception("If you are training, you have to set a config file using --cfg /path/to/your/config.yaml")
    if args.resume and args.mode == "test":
        raise Exception("args.resume is set on `test` mode: can't resume testing")
    if args.epoch is not None and args.mode == 'train':
        raise Exception("The `epoch` parameter should not be set when training")
    if args.view is not None and args.mode != "test":
        raise Exception('Visualization is only available during evaluation')
    if args.cpu:
        raise Exception("CPU training/testing is not supported: the NMS procedure is only implemented for CUDA")

    return args


def main():
    args = parse_args()
    exp = Experiment(args.exp_name, args, mode=args.mode)
    if args.cfg is None:
        cfg_path = exp.cfg_path
    else:
        cfg_path = args.cfg
    cfg = Config(cfg_path)
    exp.set_cfg(cfg, override=False)
    device = torch.device('cpu') if not torch.cuda.is_available() or args.cpu else torch.device('cuda')
    runner = Runner(cfg, exp, device, view=args.view, resume=args.resume, deterministic=args.deterministic)
    if args.mode == 'train':
        try:
            runner.train()
        except KeyboardInterrupt:
            logging.info('Training interrupted.')
    runner.eval(epoch=args.epoch or exp.get_last_checkpoint_epoch(), save_predictions=args.save_predictions)


if __name__ == '__main__':
    main()
