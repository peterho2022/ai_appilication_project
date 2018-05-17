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
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

def model_preparation():
    # 設定系統路徑
    sys.path.append("../object_detection")
    # 下載所需model
    MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
    # 類別列表
    PATH_TO_LABELS = os.path.join('object_detection/data', 'mscoco_label_map.pbtxt')
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
    return detection_graph, category_index

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def load_image():
    # For the sake of simplicity we will use only 2 images:
    # image1.jpg
    # image2.jpg
    # If you want to test the code with your images, just add path to the images to the image_paths.
    PATH_TO_TEST_IMAGES_DIR = 'object_detection/test_images'
    image_paths = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 4) ]
    # image_paths = ['test_images/image3.jpg']

    # Size, in inches, of the output images.
    image_size = (12, 8)
    return image_paths, image_size

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

def label_classes(detection_graph, image_paths):
    threshold = 0.5 # in order to get higher percentages you need to lower this number; usually at 0.01 you get 100% predicted objects
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            for image_path in image_paths:
                image = Image.open(image_path)
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                image_np = load_image_into_numpy_array(image)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
                # print ([category_index.get(value) for index,value in enumerate(classes[0]) if scores[0,index] > 0.5])
                objects = []
                for index, value in enumerate(classes[0]):
                    object_dict = {}
                    if scores[0, index] > threshold:
                        object_dict[(category_index.get(value)).get('name').encode('utf8')] = scores[0, index]
                        objects.append(object_dict)
                print(objects)
                print(len(np.where(scores[0] > threshold)[0])/num_detections[0])
    return objects

def image_label_bbox_score(image_paths, image_size):
    for image_path in image_paths:
        image = Image.open(image_path)
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        output_dict = run_inference_for_single_image(image_np, detection_graph)
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=8)
        plt.figure(figsize=image_size)
        plt.imshow(image_np)
        plt.show()

# 拍照
def photo():
    cap = cv2.VideoCapture(0)
    while(1):
        # get a frame
        ret, frame = cap.read()
        frame = cv2.resize(frame, (600,400), interpolation=cv2.INTER_CUBIC)
        # show a frame
        cv2.imshow("capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite("test.jpg", frame)
            break
    cap.release()
    cv2.destroyAllWindows()

def object_detect_stream(detection_graph, category_index):
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
                objects = []
                threshold = 0.5
                for index, value in enumerate(classes[0]):
                    object_dict = {}
                    if scores[0, index] > threshold:
                        object_dict[(category_index.get(value)).get('name').encode('utf8')] = scores[0, index]
                        objects.append(object_dict)
                print(objects)
                VideoFileOutput.write(image_np)
                cv2.imshow('live_detection',image_np)
                if cv2.waitKey(25) & 0xFF==ord('q'):
                    break
                    cv2.destroyAllWindows()
                    cap.release()

if __name__ == "__main__":
    detection_graph, category_index = model_preparation()
    object_detect_stream(detection_graph, category_index)
    # while True:
    #     photo()
    #     image_paths, image_size = load_image()
    #     objects = label_classes(detection_graph,image_paths)
    #     image_label_bbox_score(image_paths, image_size)