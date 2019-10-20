import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.datasets import mnist
import math

def normalize_data(data):
    return (data - np.mean(data))/np.max(data)

def get_filters(shape,n):
    lis = []
    for i in range(n):
        lis.append(np.random.normal(0,0.5,shape))
    return lis

def reshape_img(data):
    lis = []
    for i in data:
        lis.append(i.reshape((28,28,1)))
    return np.asarray(lis)

def convolution(img,filters):
    i_r, i_c, i_d= img.shape
    f_r, f_c ,f_d= filters[0].shape
    num_filters = len(filters)
    o_r, o_c, o_d = (i_r-f_r+1, i_c-f_c+1, num_filters)
    output_img = np.zeros((o_r, o_c, o_d))
    for d in range(o_d):
        for r in range(o_r):
            for c in range(o_c):
                output_img[r,c,d] = np.sum(filters[d] * img[r : r+f_r,c : c+f_c, :])
    return normalize_data(output_img)

def dconv_filters(error,img):
    e_r, e_c, e_d = error.shape
    i_r, i_c, i_d = img.shape
    df_r, df_c, df_d = (i_r-e_r+1, i_c-e_c+1,i_d)
    dfs = []
    for d in range(e_d):
        df = np.zeros((df_r, df_c, df_d))
        for r in range(df_r):
            for c in range(df_c):
                for img_d in range(i_d):
                    df[r,c] += np.sum(error[:,:,d] * img[r : r+e_r, c : c+e_c, img_d])
        dfs.append(df)       
    return dfs

def dconv_img(error,filters):
    e_r, e_c, e_d = error.shape
    f_r, f_c, f_d = filters[0].shape
    num_filters = len(filters)
    i_r, i_c, i_d = (e_r+f_r-1, e_c+f_c-1, f_d)
    dimg = np.zeros((i_r, i_c, i_d))
    for r in range(e_r):
        for c in range(e_c):
            for d in range(num_filters):
                dimg[r:r+f_r, c:c+f_c, :] += error[r,c,d] * filters[d]  
    return dimg

def relu(data):
    return np.where(data>0,data,0)

def relu_prime(data):
    return np.where(data>0,1,0)

def maxpool(img):
    i_r, i_c, i_d = img.shape
    f_r, f_c, = (2,2)
    o_r, o_c, o_d = (math.floor(i_r/2), math.floor(i_c/2), i_d)
    i_r, i_c, i_d = (o_r*2, o_c*2, o_d)
    out = np.zeros((o_r, o_c, o_d))
    for d in range(i_d):
        lis = []
        for r in range(0,i_r,2):
            for c in range(0,i_c,2):
                lis.append(np.max(img[r:r+2,c:c+2,d]))
        out[:,:,d] = np.asarray(lis).reshape((o_r, o_c))
    return out

def d_maxpool(img,error):
    i_r, i_c, i_d = error.shape[0]*2,error.shape[1]*2,error.shape[2]
    o_r, o_c, o_d = img.shape
    out = np.zeros((o_r, o_c, o_d))
    for d in range(i_d):
        lis = []
        for r in range(0,i_r,2):
            for c in range(0,i_c,2):
                temp = np.zeros(4)
                temp[np.argmax(img[r:r+2,c:c+2,d])] = error[int(r/2), int(c/2), d]
                out[r:r+2,c:c+2,d] = temp.reshape((2,2))
    return out

def one_hot_encoding(data):
    y_ohe = []
    for i in data:
        lis = [0]*10
        lis[i] = 1
        y_ohe.append(lis)
    return np.asarray(y_ohe)

def softmax(data):
    data = data - np.max(data)
    e = np.exp(data)
    out = e/np.sum(e)
    return out

def cross_entropy(c_output,a_output):
    return -np.sum(a_output * np.log(c_output))

def dcross_entropy(c_output,a_output):
    return c_output - a_output

def dsoftmax(data):
    return softmax(data) * (1 - softmax(data))

def update_weights(w,dw):
    for i in range(len(w)):
        w[i] = w[i]-0.01*dw[i]
    return w



(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = reshape_img(X_train)
X_test = reshape_img(X_test)
y_train = one_hot_encoding(y_train)
y_test = one_hot_encoding(y_test)

conv1_f = get_filters((3,3,1),5)
conv2_f = get_filters((3,3,5),5)
conv3_f = get_filters((3,3,5),5)
conv4_f = get_filters((3,3,5),10)

epoch = 100


for e in range(epoch):
    for x,y in zip(X_train,y_train):
        input_img = normalize_data(x)
        conv1 = convolution(input_img , conv1_f)
        maxp1 = maxpool(conv1)
        relu1 = relu(maxp1)
        # print(conv1.shape,maxp1.shape,relu1.shape)

        conv2 = convolution(relu1 , conv2_f)
        maxp2 = maxpool(conv2)
        relu2 = relu(maxp2)
        # print(conv2.shape,maxp2.shape,relu2.shape)

        conv3 = convolution(relu2 , conv3_f)
        relu3 = relu(conv3)
        # print(conv3.shape,relu3.shape)

        conv4 = convolution(relu3, conv4_f)
        output = softmax(conv4)
        # print(conv4.shape,output.shape)

        err = cross_entropy(output,y_train[5])
        # print(err)

        de_ein = dcross_entropy(output,y)
        dsoftout_softin = dsoftmax(conv4)
        de_softin = de_ein * dsoftout_softin

        de_conv4_f = dconv_filters(de_softin,relu3)
        de_conv4 = dconv_img(de_softin,conv4_f)

        de_relu3 = relu_prime(conv3) * de_conv4
        de_conv3_f = dconv_filters(de_conv4,relu2)
        de_conv3 = dconv_img(de_conv4,conv3_f)

        de_relu2 = relu_prime(maxp2)* de_conv3
        de_maxp2 = d_maxpool(conv2,de_relu2)
        de_conv2 = dconv_img(de_maxp2,conv2_f)
        de_conv2_f = dconv_filters(de_maxp2,relu1)
        
        de_relu1 = relu_prime(maxp1)*de_conv2
        de_maxp1 = d_maxpool(conv1,de_relu1)
        de_conv1_f = dconv_filters(de_maxp1,conv1)                     

        conv4_f = update_weights(conv4_f,de_conv4_f)
        conv3_f = update_weights(conv3_f,de_conv3_f)
        conv2_f = update_weights(conv2_f,de_conv2_f)
        conv1_f = update_weights(conv1_f,de_conv1_f)

        if e%2 == 0:    
            t_error = 0
            counter = 0
            for x1,y1 in zip(X_test[:100],y_test[:100]):
                counter += 1
                input_img = normalize_data(x)
                conv1 = convolution(input_img , conv1_f)
                maxp1 = maxpool(conv1)
                relu1 = relu(maxp1)
                # print(conv1.shape,maxp1.shape,relu1.shape)

                conv2 = convolution(relu1 , conv2_f)
                maxp2 = maxpool(conv2)
                relu2 = relu(maxp2)
                # print(conv2.shape,maxp2.shape,relu2.shape)

                conv3 = convolution(relu2 , conv3_f)
                relu3 = relu(conv3)
                # print(conv3.shape,relu3.shape)

                conv4 = convolution(relu3, conv4_f)
                output = softmax(conv4)
                # print(conv4.shape,output.shape)

                err = cross_entropy(output,y_train[5])
                t_error += err

            print("epoch-{}   error-{}".format(e,t_error/counter))
