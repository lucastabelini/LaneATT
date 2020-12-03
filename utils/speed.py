import time
import argparse

import torch
from thop import profile, clever_format

from lib.config import Config


def parse_args():
    parser = argparse.ArgumentParser(description="Tool to measure a model's speed")
    parser.add_argument("--cfg", default="config.yaml", help="Config file")
    parser.add_argument("--model_path", help="Model checkpoint path (optional)")
    parser.add_argument('--iters', default=100, type=int, help="Number of times to run the model and get the average")

    return parser.parse_args()


# torch.backends.cudnn.benchmark = True


def main():
    args = parse_args()
    cfg = Config(args.cfg)
    device = torch.device('cuda')
    model = cfg.get_model()
    model = model.to(device)
    test_parameters = cfg.get_test_parameters()
    height, width = cfg['datasets']['test']['parameters']['img_size']

    if args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path)['model'])

    model.eval()

    x = torch.zeros((1, 3, height, width)).to(device) + 1

    # Benchmark MACs and params

    macs, params = profile(model, inputs=(x,))
    macs, params = clever_format([macs, params], "%.3f")
    print('MACs: {}'.format(macs))
    print('Params: {}'.format(params))

    # GPU warmup
    for _ in range(100):
        model(x)

    # Benchmark latency and FPS
    t_all = 0
    for _ in range(args.iters):
        t1 = time.time()
        model(x, **test_parameters)
        t2 = time.time()
        t_all += t2 - t1

    print('Average latency (ms): {:.2f}'.format(t_all * 1000 / args.iters))
    print('Average FPS: {:.2f}'.format(args.iters / t_all))


if __name__ == '__main__':
    main()
