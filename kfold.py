import tensorflow as tf
import numpy as np
import cv2
import sys
import os
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras import regularizers
import tensorflow_hub as hub
from tensorflow.keras.layers import Flatten,Dense,Reshape,LSTM,TimeDistributed,BatchNormalization,MaxPooling1D,ReLU,Dropout,Input,Concatenate,Bidirectional
import json
import random
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import CSVLogger
#tf.keras.backend.clear_session()
#tf.config.optimizer.set_jit(True)

#location='/home/mohit/akshay/dataset/'
location='/media/mohit/faces/'
batch_size=32
threshold=0.5
epochs=20
lr=0.00037
l2=0.108

class dataGenerator(tf.keras.utils.Sequence):
	def __init__(self,_location,vid_list,batch_size):
		self.location = _location
		self.batch_size = batch_size
		self.vid_list = vid_list
		self.indexes = np.arange(len(self.vid_list))
		print(len(vid_list))

	def __getitem__(self,i):
		start = i * self.batch_size
		end = (i+1) * self.batch_size
		videos_batch = []
		labels_batch = []
		for j in range(start,end):
			item_name = self.vid_list[j]
            
			if (True):
				buff = np.load(self.location+item_name)
				if(item_name.split('.')[1] == "0"):
					labels_batch.append(0)
				else:
					labels_batch.append(1)
				videos_batch.append(buff)

		return (np.array(videos_batch), np.array(labels_batch))
    
	def __len__(self):
		return len(self.indexes) // self.batch_size

def model():
    video_input=Input(shape=(3072,16,7,7))
    x = TimeDistributed(Flatten())(video_input)
    x = MaxPooling1D(32)(x)
    x = Flatten()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.65)(x)
    #x= tf.keras.layers.MaxPooling1D()(x)
    #x = LSTM(128,activation='relu')(x)
    #x = Dropout(0.5)(x)
    x = Dense(512,activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.65)(x)
    x = Dense(32,activation='relu')(x)
    x = BatchNormalization()(x)
    output = Dense(1,activation='sigmoid')(x)
    #print(output.shape)
    model=tf.keras.Model(inputs=video_input,outputs=output)
    return model

model=model()
model.summary()


item_list=os.listdir(location)
random.shuffle(item_list)




kf = KFold(n_splits=5, shuffle=True)
fold = 1
for train_index, val_index in kf.split(item_list):
    print(f"Training on Fold {fold}")
    csv_logger = CSVLogger(f"outputs/training{fold}.csv", separator=',', append=True)
    callbacks = [
#    keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=2, verbose=1, mode='auto'),
#      keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
#                              patience=2, min_lr=0.000001, verbose=1),
#      keras.callbacks.ModelCheckpoint(filepath = 'outputs/model_.{epoch:02d}-{val_loss:.6f}.h5', verbose=1, save_best_only=True, save_weights_only = True),
      csv_logger
    ]
    train_list=[item_list[i] for i in train_index]
    val_list=[item_list[i] for i in val_index]
    steps_per_epoch=len(train_list)//batch_size
    val_steps_per_epoch=len(val_list)//batch_size

    trainDataGen=dataGenerator(location,train_list,batch_size)
    valDataGen=dataGenerator(location,val_list,batch_size)



    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr,decay=l2),loss="binary_crossentropy",metrics=[tf.keras.metrics.AUC(curve='PR'),tf.keras.metrics.BinaryAccuracy(threshold=threshold ),tf.keras.metrics.Recall(thresholds=threshold),tf.keras.metrics.Precision(thresholds=threshold)])


    history=model.fit(trainDataGen,steps_per_epoch=steps_per_epoch,validation_data=valDataGen,validation_steps=val_steps_per_epoch,epochs=epochs,callbacks=callbacks)


    model.save(f"outputs/semifinal_model{fold}")

    fig1 = plt.figure()
    plt.plot(history.history['binary_accuracy'])
    plt.plot(history.history['val_binary_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train_acc','val_acc'], loc='best')

    fig1.savefig(f"outputs/semi_accuracy{fold}.png")


    fig2 = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss','val_loss'], loc='best')

    fig2.savefig(f"outputs/semi_loss{fold}.png")


    fold += 1

