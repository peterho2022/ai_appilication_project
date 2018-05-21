# import the necessary packages
from __future__ import print_function
from PIL import Image
from PIL import ImageTk
import tkinter as tki
import threading
import datetime
import imutils
import cv2
import os

import numpy as np
import random
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


class PhotoBoothApp:
    def __init__(self, vs, outputPath):
        # store the video stream object and output path, then initialize
        # the most recently read frame, thread for reading frames, and
        # the thread stop event
        
        self.detection_graph , self.category_index = model_preparation()
        
        self.vs = vs
        self.outputPath = outputPath
        self.frame = None
        self.thread = None
        self.stopEvent = None
        
        # initialize the root window and image panel
        self.root = tki.Tk()
        self.root.geometry('800x800')
        self.panel = None
        # create a button, that when pressed, will take the current
        # frame and save it to file
        btn = tki.Button(self.root, text="Hunting !",command=self.takeSnapshot)
        btn.pack(side="bottom", fill="both", expand="yes", padx=5,pady=5)
        
#        l = tki.Label(window, 
#            text='OMG! this is TK!',    # 标签的文字
#            bg='green',     # 背景颜色
#            font=('Arial', 12),     # 字体和字体大小
#            width=15, height=2  # 标签长宽
#            )
#        l.pack()    # 固定窗口位置

        
        # start a thread that constantly pools the video sensor for
        # the most recently read frame
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.start()
    
        # set a callback to handle when the window is closed
        self.root.wm_title("PyImageSearch PhotoBooth")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)
        
        
    def check(self):
        img = cv2.imread(os.getcwd()+'\\pyimagesearch\\F4522_3.jpg',1)
        image = Image.fromarray(img)
        image = ImageTk.PhotoImage(image)
        time.sleep(10)
            
    
    def videoLoop(self):
        # DISCLAIMER:
        # I'm not a GUI developer, nor do I even pretend to be. This
        # try/except statement is a pretty ugly hack to get around
        # a RunTime error that Tkinter throws due to threading
        
        goal_object = bytes(question(), encoding = "utf8")
        try:
            # keep looping over frames until we are instructed to stop
            while not self.stopEvent.is_set():
                
                with self.detection_graph.as_default():
                    with tf.Session(graph=self.detection_graph) as sess:
                        while(1):
                            # grab the frame from the video stream and resize it to
                            # have a maximum width of 300 pixels
                            self.frame = self.vs.read()
                            self.frame = imutils.resize(self.frame, width=800,height=700)
                		
                            # OpenCV represents images in BGR order; however PIL
                            # represents images in RGB order, so we need to swap
                            # the channels, then convert to PIL and ImageTk format
                            
                            image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                            #image = Image.fromarray(image)
                            
                            
                            # 定義detection_graph的輸入和輸出向量
                            image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                            # 每個框代表檢測到特定物件
                            detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                            # 每個分數代表物件的可信度
                            # 分數和類別標籤示在圖像上
                            detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                            detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                            num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')       
                            # 由於模型可能會具有shape，因此擴展成[1, None, None, 3]
                            image_np_expanded = np.expand_dims(image, axis=0)
                            # 開始預測
                            (boxes, scores, classes, num) = sess.run(
                                [detection_boxes, detection_scores, detection_classes, num_detections],
                                feed_dict={image_tensor: image_np_expanded})
                            # 可視化預測的結果.
                            vis_util.visualize_boxes_and_labels_on_image_array(
                                image,
                                np.squeeze(boxes),
                                np.squeeze(classes).astype(np.int32),
                                np.squeeze(scores),
                                self.category_index,
                                use_normalized_coordinates=True,
                                line_thickness=8)
                            objects = []
                            threshold = 0.5
                            for index, value in enumerate(classes[0]):
                                object_dict = {}
                                if scores[0, index] > threshold:
                                    object_dict[(self.category_index.get(value)).get('name').encode('utf8')] = scores[0, index]
                                    objects.append(object_dict)
                            print(objects)
                            
                            
                            image = Image.fromarray(image)
                            
                            image = ImageTk.PhotoImage(image)
                		
                            try:
                                if list(objects[0].keys())[0] == goal_object:
                                    print("good")
                                    # 這邊我先使用睡眠 1 秒，到時候設計 GUI 的人可以改成完成特效
                                    self.check()
        
                            except IndexError:
                                pass
                            
                            
                            # if the panel is not None, we need to initialize it
                            if self.panel is None:
                                self.panel = tki.Label(image=image)
                                self.panel.image = image
                                self.panel.pack(side="left", padx=10, pady=10)
                		
                            # otherwise, simply update the panel
                            else:
                                self.panel.configure(image=image)
                                self.panel.image = image
            
        except RuntimeError:
            print("[INFO] caught a RuntimeError")

    def takeSnapshot(self):
        pass

    def onClose(self):
        # set the stop event, cleanup the camera, and allow the rest of
        # the quit process to continue
        print("[INFO] closing...")
        self.stopEvent.set()
        self.vs.stop()
        self.root.quit()
        
        
    # 載入模型
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



def question():
     goal_object = random.choice ( ['backpack', 'umbrella', 'suitcase', 'sports ball' , 'baseball bat', 'tennis racket', 'bottle'
     , 'cup', 'fork', 'knife', 'spoon', 'bowl', 'chair', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'book'
     , 'clock', 'scissors', 'toothbruth'] )
     print (goal_object)
     return goal_object
 


        
