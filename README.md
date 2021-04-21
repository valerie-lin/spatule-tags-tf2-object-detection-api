# spatule-tags-tf2-object-detection-api

## Important links
* TF object detction api: https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/index.html
* TF repo: https://github.com/tensorflow/tensorflow
* LabelImg: https://github.com/tzutalin/labelImg
* cocoAPI: https://github.com/cocodataset/cocoapi

## Prerequisites
* requirements.txt (numpy version, < 1.19, tf version 2.4 ...etc, packages versions)
* python 3.6.9 (with venv)
* GPU 1080 Ti, k20c
* labelImg: for hand labeling datasets

## Data & problem explanation
In the south of France, and in other collaborating European countries, scientists are studying birds called spatules. To better understand their behaviours such as their migrations, they have attached tags on their legs with a unique identification sequence of characters. They have also set cameras on the field which take pictures regularly (every 10 minutes). Until now, they have extracted data from these images by hand. 

The objective of the project is to perform object detection to locate the tags on images. An important second step will be to read the tags, but it is not tackled here.

The tags have a common pattern. They are made of 4 black letters or digits on a white background, with a specific font. Some letters and digits are removed to avoid confusion between similar ones (for example E is removed because it looks too much like F).
[include font here]
The main problem we faced is that there are too few tags in real data. Most of the time, birds in the pictures do not have a tag, or it is hidden by other birds or because they are sitting. We also didn’t want to hand-label thousands of images.



## Synthetic data generation

To overcome this issue, we proceeded to synthetic data generation. To do so, we had explored various techniques, such as making completely artificial tags or alpha blending real tags on other images. In the end, the best way we found is to use seamless cloning.

To do so, we still had to hand-label some images. We used the labelImg library to annotate some images. From there, we extracted some real tags. We combined them with images from our annotated images which we know do not contain tags, referred as background. The combining technique is called seamless cloning and is implemented in Python's OpenCV library. It basically allowed us to “paste” real tags on background smoothly by masking all background of the real tags image.

(include example below).

We choose a random location to paste the tag on the background so it is easy to automatically create annotations in the XML format (Pascal VOC format). The original images’ sizes are about 4000x3000. However, the object detection model we will use takes 640x640 images as inputs. So, we crop the background to match this size. The tags’ sizes are not changed.

Another issue we faced, is that we did not want the model to recognize the letters and digits in the tags but the general pattern of tags. To make sure of it in the evaluation phase, we separate tags from the training and testing dataset.

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
* To set the batch_size:
```
train_config {
  batch_size: 1
  ...
  }

```

* To set the checkpoints folder:
```
  fine_tune_checkpoint: "./pre-trained-models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0"
```


* If there is only one class on label_map.pbtxt, set the type as "detection" 
```
  fine_tune_checkpoint_type: "detection"

```
* 

include sample training


## Results/Evaluation
custom evaluation process
In pipeline.config
```
batch_size: ..
```
## Future work
