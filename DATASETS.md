## Datasets
We recommend to setup the datasets in `$LANEATT_ROOT/datasets` (it can be a symlink), where `$LANEATT_ROOT` is the code's root directory. All the configs provided in this repository expect the datasets to be in this path and changing it will require you to update the configs accordingly.
#### TuSimple
[\[Website\]](https://github.com/TuSimple/tusimple-benchmark/tree/master/doc/lane_detection)
[\[Download page\]](https://github.com/TuSimple/tusimple-benchmark/issues/3)

**How to set up**

Inside the code's root directory, run the following:
```bash
mkdir datasets # if it does not already exists
cd datasets
# train & validation data (~10 GB)
mkdir tusimple
wget "https://s3.us-east-2.amazonaws.com/benchmark-frontend/datasets/1/train_set.zip"
unzip train_set.zip -d tusimple
# test images (~10 GB)
mkdir tusimple-test
wget "https://s3.us-east-2.amazonaws.com/benchmark-frontend/datasets/1/test_set.zip"
unzip test_set.zip -d tusimple-test
# test annotations
wget "https://s3.us-east-2.amazonaws.com/benchmark-frontend/truth/1/test_label.json" -P tusimple-test/
cd ..
```
#### CULane

[\[Website\]](https://xingangpan.github.io/projects/CULane.html)
[\[Download page\]](https://drive.google.com/open?id=1mSLgwVTiaUMAb4AVOWwlCD5JcWdrwpvu)

**How to set up**

Inside the code's root directory, run the following:
```bash
mkdir datasets # if it does not already exists
cd datasets
mkdir culane
# train & validation images (~30 GB)
gdown "https://drive.google.com/uc?id=1AQjQZwOAkeBTSG_1I9fYn8KBcxBBbYyk"
gdown "https://drive.google.com/uc?id=1PH7UdmtZOK3Qi3SBqtYOkWSH2dpbfmkL"
gdown "https://drive.google.com/uc?id=14Gi1AXbgkqvSysuoLyq1CsjFSypvoLVL"
tar xf driver_23_30frame.tar.gz
tar xf driver_161_90frame.tar.gz
tar xf driver_182_30frame.tar.gz
# test images (~10 GB)
gdown "https://drive.google.com/uc?id=1LTdUXzUWcnHuEEAiMoG42oAGuJggPQs8"
gdown "https://drive.google.com/uc?id=1daWl7XVzH06GwcZtF4WD8Xpvci5SZiUV"
gdown "https://drive.google.com/uc?id=1Z6a463FQ3pfP54HMwF3QS5h9p2Ch3An7"
tar xf driver_37_30frame.tar.gz
tar xf driver_100_30frame.tar.gz
tar xf driver_193_90frame.tar.gzt
# all annotations (train, val and test)
gdown "https://drive.google.com/uc?id=1QbB1TOk9Fy6Sk0CoOsR3V8R56_eG6Xnu"
tar xf annotations_new.tar.gz
gdown "https://drive.google.com/uc?id=18alVEPAMBA9Hpr3RDAAchqSj5IxZNRKd"
tar xf list.tar.gz
```
#### LLAMAS
[\[Website\]](https://unsupervised-llamas.com/llamas/)
[\[Download page\]](https://unsupervised-llamas.com/llamas/login/?next=/llamas/download)

An account in the website is required to download the dataset.

**How to set up**
1. Download the set of color images (`color_images.zip`, 108 GB)
2. Download the annotations (`labels.zip`, 650 MB)
3. Unzip both files (`color_images.zip` and `labels.zip`) into the same directory (e.g., `datasets/llamas/`), which will be the dataset's root. This should result in a directory structure like that:
```
llamas
├── color_images
│   ├── test
│   ├── train
│   └── valid
└── labels
    ├── train
    └── valid
```

