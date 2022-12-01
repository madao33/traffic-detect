import numpy as np
import cv2
import os
from keras.models import load_model
import tensorflow as tf
# 动态分配显存，防止显存爆满
physical_device = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_device[0], True)

script_dir = os.path.dirname(__file__)
class imageTransform():

    def __init__(self, model_path = script_dir.split('codes')[0] + "/models/model480.h5"):
        self.derain_model = load_model(model_path)
        

    def equalizeColor(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.equalizeHist(v)
        hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return img

    def gamma(self, img, r):
        return np.power(img/float(255), r)

    def derain(self, img, image_shape=(720, 480)):
        img = img.reshape([1, image_shape[1], image_shape[0], 3])/255
        predict = np.asarray(self.derain_model(img))
        predict = predict.reshape([image_shape[1], image_shape[0], 3])*255
        return predict
    