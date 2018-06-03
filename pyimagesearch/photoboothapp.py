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
        self.begin = False
        # create a button, that when pressed, will take the current
        # frame and save it to file
#        btn = tki.Button(self.root, text="Hunting !",command=self.takeSnapshot)
#        btn.pack(side="bottom", fill="both", expand="yes", padx=5,pady=5)
        self.var = tki.StringVar()
        
        self.time = tki.StringVar()
        
        
        self.ques1= tki.StringVar()
        self.ques2= tki.StringVar()
        
        self.score1=tki.StringVar()
        self.score2=tki.StringVar()
        
        
        self.var.set('Hunting..')
        self.time.set('30')
        
        self.score1.set('0')
        self.score2.set('0')
        
        # start a thread that constantly pools the video sensor for
        # the most recently read frame
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.start()
    
        # set a callback to handle when the window is closed
        self.root.wm_title("PyImageSearch PhotoBooth")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)
        
        btn = tki.Button(self.root, text="start!",command=self.startgame ,font=('Arial', 12),width=10, height=2 )
        btn.place(x=700, y=700, anchor='n')    
        
    def startgame(self):
        self.begin = True

    def videoLoop(self):
        # DISCLAIMER:
        # I'm not a GUI developer, nor do I even pretend to be. This
        # try/except statement is a pretty ugly hack to get around
        # a RunTime error that Tkinter throws due to threading
        

        
        try:
            # keep looping over frames until we are instructed to stop
            while not self.stopEvent.is_set():
                with self.detection_graph.as_default():
                    with tf.Session(graph=self.detection_graph) as sess:
                        
                        while self.begin==False:
                            continue
                        
                        # set questions
                        goal_object1 = bytes(question(), encoding = "utf8")
                        goal_object2 = bytes(question(), encoding = "utf8")
                        tStart = time.time()+5    # timer start 5s for warm up
                        
                        while(1):
                            # time remain
                            tEnd = time.time()
                            remain_time= round(60-(tEnd-tStart))
                            
                            # check end game
                            if remain_time<0:
                                self.begin = False
                                break
                            
                            # check repeat task
                            while(goal_object1==goal_object2):
                                goal_object1 = bytes(question(), encoding = "utf8")
                            
                            # for finding check
                            if self.var.get()!='Hunting..':
                                time.sleep(1)
                            
                            # set questions 
                            self.ques1.set(goal_object1)
                            self.ques2.set(goal_object2)
                            
                            #set time 
                            self.time.set(remain_time)
                            
                            # showing state 
                            f = tki.Label(self.root,textvariable=self.var, font=('Arial', 18),width=15, height=2  )
                            #f.pack(side="bottom", fill="both", expand="yes", padx=5,pady=5)    # 固定窗口位置
                            f.place(x=380, y=700, anchor='n')    
                            # showing time 
                            g = tki.Label(self.root,textvariable=self.time, font=('Arial', 15),width=10, height=2  )
                            g.place(x=380, y=10, anchor='n')    
                            
                            # question　& score
                            q1 = tki.Label(self.root,textvariable=self.ques1, font=('Arial', 15),width=10, height=2  )
                            q1.place(x=70, y=10, anchor='nw')
                            
                            q2 = tki.Label(self.root,textvariable=self.ques2, font=('Arial', 15),width=10, height=2  )
                            q2.place(x=500, y=10, anchor='nw')
                            
                            s1= tki.Label(self.root,textvariable=self.score1, font=('Arial', 15),width=10, height=2  )
                            s1.place(x=220, y=10, anchor='nw')
                            
                            s2 = tki.Label(self.root,textvariable=self.score2, font=('Arial', 15),width=10, height=2  )
                            s2.place(x=650, y=10, anchor='nw')
                                        
                            
                            self.var.set('Hunting..')
                            
                            # -------------------------pattern recognition ------------------------
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
                            
                            # transform numpy into image
                            image = Image.fromarray(image)
                            image = ImageTk.PhotoImage(image)
                            # ------------------------------------------------------



                            # if match check whick team finish and set score 
                            try:
                                if (list(objects[0].keys())[0] == goal_object1) | (list(objects[0].keys())[0] == goal_object2):
                                    
                                    if list(objects[0].keys())[0] == goal_object1:
                                        self.score1.set(str(int(self.score1.get())+1))
                                        self.var.set('Team1 Founded !!')
                                        goal_object1 = bytes(question(), encoding = "utf8")
                                    else:
                                        self.score2.set(str(int(self.score2.get())+1))
                                        self.var.set('Team2 Founded !!')
                                        goal_object2 = bytes(question(), encoding = "utf8")
                                        
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
 


        
