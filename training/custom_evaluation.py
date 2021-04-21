
#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import matplotlib.pyplot as plt
import cv2
import xml.etree.ElementTree as ET
from PIL import Image
import pandas as pd

import tensorflow as tf
from object_detection.builders import model_builder
from object_detection.utils import config_util

import time
import os

# @tf.function
def detect_fn(image, model):
    """Detect objects in image."""

    image, shapes = model.preprocess(image)
    prediction_dict = model.predict(image, shapes)
    detections = model.postprocess(prediction_dict, shapes)

    return detections

def load_model(MODEL_DIR):
    print('Loading model... ', end='')
    start_time = time.time()
    tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)


    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(os.path.join(MODEL_DIR, "pipeline.config"))
    model_config = configs['model']
    detection_model = model_builder.build(model_config=model_config, is_training=False)

    # Restore checkpoint
    LAST_CKPT_NAME = sorted([fn for fn in os.listdir(MODEL_DIR) if "ckpt" in fn])[-1][:-6]
    print(LAST_CKPT_NAME)
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(MODEL_DIR, LAST_CKPT_NAME)).expect_partial()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))
    return detection_model


# In[12]:


model = load_model("./models/my_ssd_resnet50_v1_fpn")


# In[27]:


def IoU(gt_bboxes, det_bboxes, shape):
    
    gt = np.zeros(shape)
    det = np.zeros(shape)
    
    for x1,y1,x2,y2 in gt_bboxes:
        gt[y1:y2, x1:x2] = 1
        
    for x1,y1,x2,y2 in det_bboxes:
        det[y1:y2, x1:x2] = 1
    
    gt_area = np.sum(gt)
    det_area = np.sum(det)
    union = np.sum((gt+det)>0)
    inter = np.sum((gt+det)>1)
    if union == 0: return 1., gt_area, det_area
    return inter/union, gt_area, det_area


# In[14]:


def get_bboxes_from_xml(xml_file):
    bboxes = []
    annotation = ET.parse(xml_file)
    if annotation.find("object") is not None:
        for bbox in annotation.find("object").findall("bndbox"):
            xmin = int(bbox.find("xmin").text)
            xmax = int(bbox.find("xmax").text)
            ymin = int(bbox.find("ymin").text)
            ymax = int(bbox.find("ymax").text)
            bboxes.append((xmin, ymin, xmax, ymax))
    return bboxes


# In[15]:


def get_bboxes_from_detection(img_file, model, score_threshold):
    bboxes = []
    image = Image.open(img_file)
    image_tensor = tf.convert_to_tensor(np.expand_dims(image, 0), dtype=tf.float32)
    detections = detect_fn(image_tensor, model)
    for i, score in enumerate(detections["detection_scores"][0]):
        if score > score_threshold:
            bbox = detections["detection_boxes"][0][i].numpy() * 640
            bbox = list(bbox.astype(int))
            bbox[1], bbox[0], bbox[3], bbox[2] = bbox
            bboxes.append(bbox)
    return bboxes


# In[29]:


def run_eval(eval_folder):
    images = [os.path.join(eval_folder, fn) for fn in os.listdir(eval_folder) if fn[-4:].lower()==".jpg" and "._" not in fn.lower()]
    cols = ["image", "Gt_n_tags", "Gt_bboxes"]
    thresholds = [0.3, 0.7]
    for t in thresholds:
        cols.append("IoU_"+str(t))
        cols.append("Gt_area_"+str(t))
        cols.append("Det_area_"+str(t))
        cols.append("Det_n_tags_"+str(t))
        cols.append("Det_bboxes_"+str(t))
    metrics = []

    for i, image in enumerate(images):
        gt_boxes = get_bboxes_from_xml(image[:-4]+".xml")
        image_metrics = [image, len(gt_boxes), str(gt_boxes)]
        for t in thresholds:

            det_boxes = get_bboxes_from_detection(image, model, t)
            (iou, gt_area, det_area) = IoU(gt_boxes, det_boxes, (640,640))
            image_metrics.append(iou)
            image_metrics.append(gt_area)
            image_metrics.append(det_area)
            image_metrics.append(len(det_boxes))
            image_metrics.append(str(det_boxes))
            
        metrics.append(image_metrics)
        print(" "*60, end="\r") #clean prev line
        print('Finished {}/{}. IoU: {}'.format(i+1, len(images), iou), end="\r")
    metrics = pd.DataFrame(data=metrics, columns=cols)
    return metrics
m = run_eval("./images/test/")
m.to_csv("./eval_metrics_test.csv")

