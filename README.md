# spatule-tags-tf2-object-detection-api

## Important links
* TF object detction api: https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/index.html
* TF repo: https://github.com/tensorflow/tensorflow
* LabelImg: https://github.com/tzutalin/labelImg
* cocoAPI:https://github.com/cocodataset/cocoapi

## Prerequisites
* requirements.txt (numpy version, < 1.19, tf version 2.4 ...etc, packages versions)
* python 3.6.9 (with venv)
* GPU 1080 Ti, k20c
* labelImg: for hand labeling daasets

## Data & problem explanation
include sample data
tags description (letters include font etc)


## Synthetic data creation
include sample synth data
seamless cloning
separating eval and train tags 

## Training

The structure should be like:
```
.
├── annotations
│   ├── label_map.pbtxt
│   ├── test.record
│   └── train.record
├── exported-models
├── images
│   ├── test
│   └── train
├── model_main_tf2.py
├── models
│   └── my_ssd_resnet50_v1_fpn
│       └── pipeline.config
└── pre-trained-models
    └── ssd_resnet50_v1_fpn_640x640_coco17_tpu-8
        ├── checkpoint
        │   ├── checkpoint
        │   ├── ckpt-0.data-00000-of-00001
        │   └── ckpt-0.index
        ├── pipeline.config
        └── saved_model
            ├── assets
            ├── saved_model.pb
            └── variables
                ├── variables.data-00000-of-00001
                └── variables.index

```
In the `/images/train` and `/images/test` files, we can add our .jpg and the corresponding xml files.


In pipeline.config
```
steps ... 
evaluation folder... 

```

include sample training


## Results/Evaluation
custom evaluation process
In pipeline.config
```
batch_size: ..
```
## Future work
