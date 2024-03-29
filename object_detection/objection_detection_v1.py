# -*- coding: utf-8 -*-
'''
@author: Chia Yu, Ho
@date: 20180430
'''
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

import cv2

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util

# 下載所需model
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

cap=cv2.VideoCapture(0) # 0 代表與第一個攝影機連接
filename="output0.avi"
codec=cv2.VideoWriter_fourcc('m','p','4','v')# fourcc代表四個字符代碼
framerate=30
resolution=(640,480)
    
VideoFileOutput=cv2.VideoWriter(filename,codec,framerate, resolution)
    
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:  
    ret=True
    while (ret):
        ret, image_np=cap.read()
        # 定義detection_graph的輸入和輸出向量
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # 每個框代表檢測到特定物件
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # 每個分數代表物件的可信度
        # 分數和類別標籤示在圖像上
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')       
          # 由於模型可能會具有shape，因此擴展成[1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
          # 開始預測
        (boxes, scores, classes, num) = sess.run(
              [detection_boxes, detection_scores, detection_classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})
          # 可視化預測的結果.
        vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=8)
  
        VideoFileOutput.write(image_np)
        cv2.imshow('live_detection',image_np)
        if cv2.waitKey(25) & 0xFF==ord('q'):
            break
            cv2.destroyAllWindows()
            cap.release()