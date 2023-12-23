import tensorflow as tf
import numpy as np
import cv2
from facenet_pytorch import MTCNN
import sys
import os
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras import regularizers
import tensorflow_hub as hub
from tensorflow.keras.layers import Flatten,Dense,Reshape,LSTM,TimeDistributed,BatchNormalization,MaxPooling2D,Dropout,Input,Concatenate
import json
import random
from PIL import Image

#tf.keras.backend.clear_session()
#tf.config.optimizer.set_jit(True)

root_directory = '/media/mohit/'


batch_size=2

model_url="https://tfhub.dev/shoaib6174/swin_base_patch244_window877_kinetics400_22k/1"
base_model=hub.KerasLayer(model_url,trainable=False)
mtcnn=MTCNN(keep_all=True)





class dataGenerator(tf.keras.utils.Sequence):
    def __init__(self,_location,vid_list,batch_size):
        self.location = _location
        self.batch_size = batch_size
        self.vid_list = vid_list
        self.indexes = np.arange(len(self.vid_list))
        with open(os.path.join(self.location,'metadata.json'), 'r') as json_file:
            self.data = json.load(json_file)
        print(len(vid_list),"------>>>>>>>>>")
        
    def __getitem__(self,i):
        start = i * self.batch_size
        end = (i+1) * self.batch_size
        videos_batch = []
        labels_batch = []
        
            
        for j in range(start,end):
        
            item = self.vid_list[j]
            cap=cv2.VideoCapture(os.path.join(self.location,item))
            buff=np.zeros((32,224,224,3))
            nFrames=0
            i=0
            while True:
                ret,frame=cap.read()

                if not ret:
                    print("cannot read----------------")
                    flag=True
                    break
                if(nFrames%7==0 and i<32):
                    faces=mtcnn(frame[:, :, ::-1])
                    try:
	                    face=np.transpose(faces[0].numpy(),(1,2,0))
	                    face=np.array((face*127.0)+128.0,dtype=np.uint8)
	                    face=np.array(Image.fromarray(face).resize((224, 224)))
	                    buff[i]=face/255.0
	                    i+=1
                        
                    except:
                        print("******failed_here*********")
                    
                elif i>=32:
                    break
                nFrames+=1
            cap.release()
            


            if(self.data[item]["label"]=="FAKE"):
                labels_batch.append(1)
            else:
                labels_batch.append(0)
            videos_batch.append(np.transpose(buff,(3,0,1,2)))

        return (np.array(videos_batch), np.array(labels_batch))
    
    def __len__(self):
        return len(self.indexes) // self.batch_size
count=0
for dir in os.listdir(root_directory):
    location = os.path.join(root_directory, dir)

    if os.path.isdir(location) and dir.startswith('dfdc_train_part_'):
        

        item_list=os.listdir(location)
        item_list.remove('metadata.json')
        random.shuffle(item_list)

        trainDataGen=dataGenerator(location,item_list,batch_size)

        for data,label in trainDataGen:
#            nFrame=0
            print(count,label[0],label[1])
#            base_out=None
#            for _ in range(3):
#                if base_out is None:
            base_out=base_model(data)
#                else:
#                    base_out=Concatenate(axis=1)([base_out,base_model(data[:,:,nFrame:nFrame+32,:,:])])
#                nFrame += 32
            np.save("/media/mohit/faces/"+str(count)+"."+str(label[0])+".npy",base_out[0])
            np.save("/media/mohit/faces/"+str(count+1)+"."+str(label[1])+".npy",base_out[1])
            count+=2



