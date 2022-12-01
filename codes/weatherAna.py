import cv2
import numpy as np
import pickle
import sys
sys.path.append('..')
import os
script_dir = os.path.dirname(__file__)
class weatherAna():

    def __init__(self, model_path = script_dir.split('codes')[0] + "/models/xgb_clf.pickle.dat"):
        self.weather_model = pickle.load(open(model_path, 'rb'))

    def darkVector(self, img, m=10, n=10):
        """
            获取图像雾度特征
            @ param: 
                img-输入的图像
                m: 分块的宽度，默认为10
                n: 分块的长度，默认为10
            @ return:
                返回值为mxn维的暗通道向量
        """
        # 获取图像的基本信息
        height, width, channel = img.shape
        grid_h = int(height*1.0/(m-1)+0.5)
        grid_w = int(width*1.0/(n-1)+0.5)

        # 满足整除关系的高、宽
        h=grid_h*(m-1)
        w=grid_w*(n-1)
        # 图像缩放
        img_re = cv2.resize(img, (w, h), cv2.INTER_LINEAR)
        gx, gy = np.meshgrid(np.linspace(0, w, n), np.linspace(0, h, m))
        gx = gx.astype(np.int)
        gy = gy.astype(np.int)

        dark_img = np.zeros([m, n], np.uint8)
        for i in range(m-1):
            for j in range(n-1):
                for k in range(channel):
                    # 计算分块区域的最小值
                    dark_img[i,j] += np.min(img_re[gy[i, j]: gy[i+1, j+1], gx[i, j]:gx[i+1, j+1],:])
                dark_img[i, j] /= channel # 计算各个通道平均值

        return dark_img.reshape(m*n, -1)

    def saturationVector(self, img):
        """
        获取图像饱和对比度特征
        @ params:
            img-输入的原始图像
        return:
            返回的是图像归一化饱和度的直方图，为100维特征
        """
        # bgr转换为hsv，并分离出饱和度通道s
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        height, width, col = img.shape
        # 归一化饱和度
        _range = np.max(s)-np.min(s)
        s_img = np.zeros((100,1), np.float32)
        s_img = (s-np.min(s))/_range
        s_img = s_img.astype(np.float32)
        hist = cv2.calcHist([s_img], [0], None, [100], [0, 1])
        return s_img, hist

    def getfeatures(self, img):
        _simg, hist = self.saturationVector(img)
        return np.concatenate((self.darkVector(img), hist), axis=1).reshape(1, 200)
    
    def getWeather(self, img):
        return self.weather_model.predict(self.getfeatures(img))