#coding:utf-8
import os
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2

classes={'h', 'g'}
writer=tf.python_io.TFRecordWriter('train.tfrecords')

for idx, name in enumerate(classes):
    class_path = './'+name+'/'
    for img_name in os.listdir(class_path):
        img_path = class_path + img_name
        print(img_path)

        img = Image.open(img_path)
        img = img.convert('RGB')  # img.mode默认为RGBA模式
        img = img.resize((128,128), Image.ANTIALIAS)
        img_raw = img.tobytes() # 将图片转化为二进制格式

        # img = cv2.imread(img_path)
        # img = cv2.resize(img, (128,128))
        # img_raw = img.tobytes()

        print(len(img_raw))

        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[idx])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        })) #example对象对label和image数据进行封装
        writer.write(example.SerializeToString())

writer.close()

