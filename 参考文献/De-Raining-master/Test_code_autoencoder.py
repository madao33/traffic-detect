import numpy as np
from keras.models import Sequential
from keras.layers import Input, Concatenate
from keras.layers import Conv2D, Conv2DTranspose, Activation, MaxPooling2D, UpSampling2D
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.models import Model
from keras.models import load_model
import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam
import cv2
import matplotlib.pyplot as plt
import glob
from keras import losses

'''

    PUT THIS FILE IN THE DIRECTORY HAVING YOUR TEST IMAGES IF YOUR TEST IMAGES ARE OF THE FORM AS 
    TRAINING IMAGES WERE THIS FILE WILL STRAIGHT GIVE YOU PSNR .  


'''




autoEncoder = load_model("./NNFL_FINAL_WEIGHTS")

'''

    PLEASE USE THE FOLLOWING FUNCTION TO SEND IN YOUR DATA TO TEST 
    PLEASE ENTER YOUR DATA IN FORM OF (Batch_size,256,256,3) . Example : (700,256,256,3)
    ON TRAINING SET PSNR : 21.79 
    ACCURACY USING KERAS METRICS : MORE THAN 82%
    IMAGES WERE GETTING PROPERLY DERAINED 

'''

def test(X):
    X = np.asarray(X)
    inp = X/255
    out = autoEncoder.predict(inp)
    out = out*255
    out = out.astype(np.uint8)
    return out 



'''

    PLACE THIS FILE IN YOUR FOLDER WHERE YOU HAVE ALL YOUR TEST IMAGES THIS FUNCTION READS ALL YOUR TEST IMAGES 

'''


images = []
for i in glob.glob("*.jpg"):
    images.append(cv2.imread(i))




'''

    IF IN FORM OF .csv format  uncomment these 2 lines :

    rain_im = np.genfromtxt('rain.csv',delimeter = ',')
    derain_im = np.genfromtxt('rain.csv',delimeter = ',')

    THEN YOU CAN BYPASS NEXT STEP AND GOTO rain_final and derain_final step : 

'''

'''

    IF IMAGES NEED TO BE SPLIT DO LIKE THIS : 
    ASSUMING IMAGES ARE IN images and in the form given to us for training : 

'''
co = 0
rain_im = []
derain_im = []
for i in images:
    #print(co)
    co += 1
    col = i.shape[1]
    im = i[:,:col//2,:]
    im2 = i[:,col//2:,:]
    derain_im.append(im)
    rain_im.append(im2)




'''

    RESIZING OF IMAGES DONE DURING THIS :
    AND CONVERTED TO numpy arrays :

'''
derain_final = []
for i in derain_im:
    x = cv2.resize(i,(256,256))
    derain_final.append(x)

rain_final = []
for i in rain_im:
    x = cv2.resize(i,(256,256))
    rain_final.append(x)

rain_final = np.asarray(rain_final)
derain_final = np.asarray(derain_final)


'''

    THE FOLLOWING CODE CALCULATES PSNR PLEASE USE ACCORDING TO YOUR REQUIREMENTS : 
    PASSED BATCH OF SIZE 1 : SHAPE : (1,256,256,3)

'''

a = test(rain_final[:1])

mse = ((derain_final[:1] - a) ** 2).mean(axis=None)
psnr = 20*np.log10(255/(mse**(1/2.0)))
print(psnr)




'''

THIS IS MY ENTIRE TRAINING CODE :

input_img = Input(shape=(256, 256, 3))

x = Conv2D(16, (4, 4), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(32, (5, 5), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)

encoded = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
#encoded = MaxPooling2D((2, 2), padding='same')(x)


x = Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(encoded)

x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (5, 5), activation='relu', padding='same')(x)

x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid',padding = 'same')(x)

#decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.summary()
autoencoder.compile(optimizer='adam', loss=losses.mean_squared_error,metrics = ['accuracy'])


Model.fit(autoencoder , rain_final, derain_final,epochs=250,shuffle=True,batch_size = 35)

autoencoder.save("./NNFL_FINAL_WEIGHTS")

'''