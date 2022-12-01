import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import glob

images = []
for i in glob.glob("*.jpg"):
    images.append(cv2.imread(i))

co = 0
rain_im = []
derain_im = []
for i in images:
    co += 1
    col = i.shape[1]
    im = i[:,:col//2,:]
    im2 = i[:,col//2:,:]
    derain_im.append(im)
    rain_im.append(im2)

derain_final = []
for i in derain_im:
    x = cv2.resize(i,(256,256))
    derain_final.append(x)

rain_final = []
for i in rain_im:
    x = cv2.resize(i,(256,256))
    rain_final.append(x)

k_rain = []
batch_size = 7
for i in range(0,700,batch_size):
    ldd = []
    for j in range(i,i+batch_size):
        ldd.append(rain_final[j])
    k_rain.append(ldd)

k_derain = []
for i in range(0,700,batch_size):
    ldd = []
    for j in range(i,i+batch_size):
        ldd.append(derain_final[j])
    k_derain.append(ldd)

def prelu(_x):
    if not hasattr(prelu, "alpha"):
         prelu.alpha = 0
    global _alpha
    _alpha = tf.get_variable(str(prelu.alpha), _x.get_shape()[-1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    prelu.alpha += 1
    return tf.maximum(_alpha*_x, _x)


x1 = tf.placeholder(tf.float32,shape=[None,256,256,3])
Y = tf.placeholder(tf.float32,shape=[None,256,256,3])

x2 = tf.layers.conv2d(x1,64,3,strides=1,padding='same')
x2 = tf.layers.batch_normalization(x2)
x2 = prelu(x2)

x3 = tf.layers.conv2d(x2,64,3,strides=1,padding='same')
x3 = tf.layers.batch_normalization(x3)
x3 = prelu(x3)

x4 = tf.layers.conv2d(x3,64,3,strides=1,padding='same')
x4 = tf.layers.batch_normalization(x4)
x4 = prelu(x4)

x5 = tf.layers.conv2d(x4,64,3,strides=1,padding='same')
x5 = tf.layers.batch_normalization(x5)
x5 = prelu(x5)

x6 = tf.layers.conv2d(x5,32,3,strides=1,padding='same')
x6 = tf.layers.batch_normalization(x6)
x6 = prelu(x6)

x7 = tf.layers.conv2d(x6,1,3,strides=1,padding='same')
x7 = tf.layers.batch_normalization(x7)
x7 = prelu(x7)

x8 = tf.layers.conv2d_transpose(x7,32,3,strides=1,padding='same')
x8 = tf.layers.batch_normalization(x8)
x8 = tf.nn.relu(x8)

x9 = tf.layers.conv2d_transpose(x8,64,3,strides=1,padding='same')
x9 = tf.layers.batch_normalization(x9)
x9 = tf.nn.relu(x9)

x9 = x9 + x5

x10 = tf.layers.conv2d_transpose(x9,64,3,strides=1,padding='same')
x10 = tf.layers.batch_normalization(x10)
x10 = tf.nn.relu(x10)

x11 = tf.layers.conv2d_transpose(x10,64,3,strides=1,padding='same')
x11 = tf.layers.batch_normalization(x11)
x11 = tf.nn.relu(x11)

x11 = x11 + x3

x12 = tf.layers.conv2d_transpose(x11,64,3,strides=1,padding='same')
x12 = tf.layers.batch_normalization(x12)
x12 = tf.nn.relu(x12)

x13 = tf.layers.conv2d_transpose(x12,3,3,strides=1,padding='same')
x13 = tf.layers.batch_normalization(x13)
x13 = tf.nn.relu(x13)

logits = x13 + x1





x01 = tf.layers.conv2d(Y,48,4,strides=2,padding='same')
x01 = tf.layers.batch_normalization(x01)

x02 = tf.layers.conv2d(x01,96,4,strides=2,padding='same')
x02 = tf.layers.batch_normalization(x02)
x02 = prelu(x02)

x03 = tf.layers.conv2d(x02,192,4,strides=2,padding='same')
x03 = tf.layers.batch_normalization(x03)
x03 = prelu(x03)

x04 = tf.layers.conv2d(x03,384,4,strides=1,padding='same')
x04 = tf.layers.batch_normalization(x04)
x04 = prelu(x04)

x05 = tf.layers.conv2d(x04,1,4,strides=1,padding='same')

flat = tf.reshape(x05,(-1,32*32*1))
logits2 = tf.layers.dense(flat,1)
out = tf.sigmoid(logits2)



Euclidean_loss = tf.reduce_mean(tf.squared_difference(logits,Y))



x101 = tf.layers.conv2d(logits,48,4,strides=2,padding='same')
x101 = tf.layers.batch_normalization(x101)

x102 = tf.layers.conv2d(x101,96,4,strides=2,padding='same')
x102 = tf.layers.batch_normalization(x102)
x102 = prelu(x102)

x103 = tf.layers.conv2d(x102,192,4,strides=2,padding='same')
x103 = tf.layers.batch_normalization(x103)
x103 = prelu(x103)

x104 = tf.layers.conv2d(x103,384,4,strides=1,padding='same')
x104 = tf.layers.batch_normalization(x104)
x104 = prelu(x104)

x105 = tf.layers.conv2d(x104,1,4,strides=1,padding='same')

flat2 = tf.reshape(x105,(-1,32*32*1))
logits12 = tf.layers.dense(flat2,1)
out2 = tf.sigmoid(logits12)


La_Y = -tf.reduce_mean(tf.log(out))
La_X = -tf.reduce_mean(tf.log(1-out2))
La = La_Y + La_X


loss = 0.0066*La + Euclidean_loss

solver =  tf.train.AdamOptimizer(0.002).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

epochs = 10
batches = 100

for i in range(2):
    
    print(sess.run(solver,feed_dict={x1:k_rain[0],Y:k_derain[0]}))

img=sess.run(logits,feed_dict={x1:k_rain[0],Y:k_derain[0]})
plt.imshow(img[0])
plt.show()

                                    
