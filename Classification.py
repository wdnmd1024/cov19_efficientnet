import pickle

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization, Dense, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
import pathlib

import math
import os
import keras
import tensorflow as tf
from tensorflow.keras import layers


def load_data():
    """=============== 加载数据 ==============="""
    infile = open('../input/Processed_Data.cp', 'rb')
    data_dict = pickle.load(infile)
    all_cts = data_dict['cts']
    all_inf = data_dict['infects']
    infile.close()

    from sklearn.utils import shuffle
    all_cts, all_inf = shuffle(all_cts, all_inf)  # 数据同步混洗

    all_cts = np.array(all_cts)
    all_inf = np.array(all_inf)

    print(all_cts.shape)
    print(all_inf.shape)

    # 数据标准化, 映射到0 1区间
    all_cts = (all_cts - all_cts.min()) / (all_cts.max() - all_cts.min())
    all_inf = (all_inf - all_inf.min()) / (all_inf.max() - all_inf.min())

    # print("{} {}".format(all_cts.min(), all_cts.max()))
    # print("{} {}".format(all_inf.min(), all_inf.max()))

    """=============== 创建标签 ==============="""
    total_slides = len(all_cts)
    index_arr = []
    inf_check = np.ones((len(all_inf)))
    for i in range(len(all_inf)):
        if np.unique(all_inf[i]).size == 1:
            inf_check[i] = 0
            index_arr.append(i)
    # print("Number of CTS with no infection ", len(index_arr))

    """=============== 划分数据集6:2:2 ==============="""
    X_train = all_cts[:int(len(all_cts) * 0.6)]
    Y_train = inf_check[:int(len(inf_check) * 0.6)]
    X_val = all_cts[int(len(all_cts) * 0.6):int(len(all_cts) * 0.8)]
    Y_val = inf_check[int(len(inf_check) * 0.6):int(len(inf_check) * 0.8)]
    X_test = all_cts[int(len(all_cts) * 0.8):]
    Y_test = inf_check[int(len(inf_check) * 0.8):]
    X_test_inf = all_inf[int(len(all_inf) * 0.8):]

    print("{} {}".format(X_train.shape, Y_train.shape))
    print("{} {}".format(X_val.shape, Y_val.shape))
    print("{} {}".format(X_test.shape, Y_test.shape))

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, X_test_inf



# 定义网络框架
def get_model(width=128, height=128):
    inputs = Input((width, height, 1))
    x = Conv2D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = MaxPooling2D(pool_size=2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=64, kernel_size=3, activation="relu")(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=128, kernel_size=3, activation="relu")(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=256, kernel_size=3, activation="relu")(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(units=512, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(units=1, activation="sigmoid")(x)
    # Define the model.
    model = Model(inputs, outputs, name="2dcnn")
    return model

# efficientnetv2网络结构

#Swish激活函数的实现
#网络的深度缩放
#网络的宽度缩放
import math
import os
import keras
import tensorflow as tf
from tensorflow.keras import layers

def Swish(inputs):
    return inputs*tf.sigmoid(inputs)

class MBConvxBlock(tf.keras.Model):
    def __init__(self,filter_inp,expand_rato,dropout_rate,kernel_size=(3,3),strides=(1,1)):
        super(MBConvxBlock, self).__init__()
        self.strides=strides

        #逐点卷积（1x1卷积）归一化
        self.conv11=layers.Conv2D(expand_rato*filter_inp,kernel_size=[1,1],strides=[1,1],padding='same')
        self.BN1=layers.BatchNormalization()

        #深度可分离卷积
        self.depthwise=layers.DepthwiseConv2D(kernel_size=kernel_size,strides=strides,padding='same')
        self.BN2=layers.BatchNormalization()

        #SE操作
        self.SE=keras.Sequential([
            layers.AveragePooling2D(pool_size=[1,1]),
            layers.Conv2D(filter_inp//4,kernel_size=[1,1],strides=[1,1],padding='same'),
            layers.Conv2D(filter_inp//4,kernel_size=[1,1],strides=[1,1],padding='same')
        ])

        self.conv22=layers.Conv2D(filter_inp,kernel_size=[1,1],strides=[1,1],padding='same')

        self.BN3=layers.BatchNormalization()
        #连接失活
        self.dropout=layers.Dropout(dropout_rate)

        #如果步长为1的话，残差连接
        if strides==1:
            self.shortcut=layers.Conv2D(filter_inp,kernel_size=[1,1],strides=[1,1],padding='same')
            self.shortcutBN=layers.BatchNormalization()

    def call(self,inputs,training=None):
        x=self.conv11(inputs)
        x=self.BN1(x)
        x=Swish(x)

        x=self.depthwise(x)
        x=self.BN2(x)
        x=Swish(x)

        x=self.SE(x)
        x=self.conv22(x)
        x=self.BN3(x)

        x=self.dropout(x)

        if self.strides==1:
            x1=self.shortcut(inputs)
            x1=self.shortcutBN(x1)
            x_out=tf.add(x,x1)
            return x_out
        return x

class MBConvx_FusedBlock(tf.keras.Model):
    def __init__(self,filter_inp,expand_rato,dropout_rate,kernel_size=(3,3),strides=(1,1)):
        super(MBConvx_FusedBlock, self).__init__()
        self.strides=strides


        #逐点卷积（1x1卷积）归一化
        self.conv11=layers.Conv2D(expand_rato*filter_inp,kernel_size=kernel_size,strides=strides,padding='same')
        self.BN1=layers.BatchNormalization()

        #SE操作
        self.SE=keras.Sequential([
            layers.AveragePooling2D(pool_size=[1,1]),
            layers.Conv2D(filter_inp//4,kernel_size=[1,1],strides=[1,1],padding='same'),
            layers.Conv2D(filter_inp//4,kernel_size=[1,1],strides=[1,1],padding='same')
        ])

        self.conv22=layers.Conv2D(filter_inp,kernel_size=[1,1],strides=[1,1],padding='same')

        self.BN3=layers.BatchNormalization()
        #连接失活
        self.dropout=layers.Dropout(dropout_rate)

        #如果步长为1的话，残差连接
        if strides==1:
            self.shortcut=layers.Conv2D(filter_inp,kernel_size=[1,1],strides=[1,1],padding='same')
            self.shortcutBN=layers.BatchNormalization()

    def call(self,inputs,training=None):
        x=self.conv11(inputs)
        x=self.BN1(x)
        x=Swish(x)


        x=self.SE(x)
        x=self.conv22(x)
        x=self.BN3(x)

        x=self.dropout(x)

        if self.strides==1:
            x1=self.shortcut(inputs)
            x1=self.shortcutBN(x1)
            x_out=tf.add(x,x1)
            return x_out
        return x


class EfficientNetV2(tf.keras.Model):
    def __init__(self,width_ceoff,depth_ceoff,dropout=0.2):
        super(EfficientNetV2, self).__init__()

        self.width_coeff=width_ceoff
        self.depth_ceoff=depth_ceoff
        self.divisor=8

        #输入的第一个3x3卷积操作
        self.conv11=layers.Conv2D(24,kernel_size=[3,3],strides=[2,2],padding='same')
        self.BN1=layers.BatchNormalization()

        self.MBConvblock1 = self.MBConvxFused(filter_inp=self.rounds_width(24),expand_rato=1,layers=self.rounds_depth(2), kernel_size=(3,3),  strides=(1,1), dropout_rate=0.2,i=1)
        self.MBConvblock2 = self.MBConvxFused(filter_inp=self.rounds_width(48),expand_rato=4, layers=self.rounds_depth(4), kernel_size=(3, 3), strides=(2,2), dropout_rate=0.2,i=2)
        self.MBConvblock3 = self.MBConvxFused(filter_inp=self.rounds_width(64),expand_rato=4, layers=self.rounds_depth(4), kernel_size=(5,5), strides=(2,2), dropout_rate=0.2,i=3)
        self.MBConvblock4 = self.MBConvx(filter_inp=self.rounds_width(128), expand_rato=4,layers=self.rounds_depth(6), kernel_size=(3, 3), strides=(2,2), dropout_rate=0.2,i=4)
        self.MBConvblock5 = self.MBConvx(filter_inp=self.rounds_width(160),expand_rato=6, layers=self.rounds_depth(9), kernel_size=(5,5), strides=(1,1), dropout_rate=0.2,i=5)
        self.MBConvblock6 = self.MBConvx(filter_inp=self.rounds_width(256), expand_rato=6,layers=self.rounds_depth(15), kernel_size=(5,5), strides=(2,2), dropout_rate=0.2,i=6)

        self.conv22=layers.Conv2D(1280,kernel_size=[1,1],strides=[1,1],padding='same')
        self.BN2=layers.BatchNormalization()

        self.avgpooling=layers.GlobalAveragePooling2D()
        self.dropout=layers.Dropout(dropout)
        self.dense=layers.Dense(1000)
        self.softmax=layers.Activation('softmax')


    def MBConvx(self,filter_inp,layers,expand_rato,kernel_size,strides,dropout_rate,i):
        mbconv=keras.Sequential([],name='MBConv'+str(i))
        mbconv.add(
            MBConvxBlock(filter_inp,expand_rato, dropout_rate, kernel_size, strides)
        )
        for i in range(1,layers):
            mbconv.add(
                MBConvxBlock(filter_inp,expand_rato,dropout_rate,kernel_size)
            )
        return mbconv

    def MBConvxFused(self,filter_inp,expand_rato,layers,kernel_size,strides,dropout_rate,i):
        mbconvfused=keras.Sequential([],name='MBConv'+str(i))
        mbconvfused.add(
            MBConvx_FusedBlock(filter_inp, expand_rato,dropout_rate, kernel_size, strides)
        )
        for i in range(1,layers):
            mbconvfused.add(
                MBConvx_FusedBlock(filter_inp,expand_rato,dropout_rate,kernel_size)
            )
        return mbconvfused

    #计算缩放之后的宽度
    def rounds_width(self,filters):
        filters*=self.width_coeff
        #计算之后的宽度
        new_filters=max(self.divisor,int(filters+self.divisor/2)//self.divisor*self.divisor)
        if new_filters<0.9*filters:
            new_filters+=self.divisor
        return int(new_filters)

    #计算深度
    def rounds_depth(self,layers):
        return int(math.ceil((self.depth_ceoff*layers)))

    def call(self,inputs,training=None):
        x=self.conv11(inputs)
        x=self.BN1(x)
        x=Swish(x)

        x = self.MBConvblock1(x)
        x = self.MBConvblock2(x)
        x = self.MBConvblock3(x)
        x = self.MBConvblock4(x)
        x = self.MBConvblock5(x)
        x = self.MBConvblock6(x)

        x=self.conv22(x)
        x=self.BN2(x)
        x=Swish(x)

        x=self.avgpooling(x)
        x=self.dropout(x)
        x=self.dense(x)
        x_out=self.softmax(x)

        return x_out

def efficientnet_Bx(width_ceoff=1.0,depth_ceoff=1.0,resolution=224,dropout=0.2):
    efficientnetV2S = EfficientNetV2(width_ceoff=width_ceoff, depth_ceoff=depth_ceoff, dropout=dropout)
    efficientnetV2S.build(input_shape=(None,resolution,resolution, 3))
    efficientnetV2S.summary()

    return efficientnetV2S

if __name__ == '__main__':
    print('Pycharm')
    efficientnet_Bx(width_ceoff=1.0,depth_ceoff=1.0,resolution=224,dropout=0.2)




if __name__ == '__main__':
    # 加载模型
    #model = get_model(width=128, height=128)

    model = efficientnet_Bx(width_ceoff=1.0, depth_ceoff=1.0, resolution=224, dropout=0.2)

    # print(model.summary())
    # 编译模型
    initial_learning_rate = 0.0001  # 学习率
    lr_schedule = optimizers.schedules.ExponentialDecay(  # 优化器
        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
    )
    model.compile(
        loss="binary_crossentropy",  # 交叉熵损失函数
        optimizer=optimizers.Adam(learning_rate=lr_schedule),  # Adam优化器
        metrics=["acc"],
    )
    # 在每个训练期之后保存模型，只保存最好的
    checkpoint_cb = callbacks.ModelCheckpoint(
        "../output/3d_image_classification.h5", save_best_only=True
    )

    # 加载数据集
    X_train, Y_train, X_val, Y_val, X_test, Y_test, X_test_inf = load_data()

    # Training
    path = pathlib.Path("../output/3d_image_classification.h5")
    if not path.exists():
        history = model.fit(X_train, Y_train, epochs=100, validation_data=(X_val, Y_val), callbacks=[checkpoint_cb],
                            shuffle=True, verbose=1)

        # Plotting the accuracy and loss
        fig, ax = plt.subplots(1, 2, figsize=(20, 5))
        ax = ax.ravel()

        for i, metric in enumerate(["acc", "loss"]):
            ax[i].plot(history.history[metric])
            ax[i].plot(history.history["val_" + metric])
            ax[i].set_title("Model {}".format(metric))
            ax[i].set_xlabel("epochs")
            ax[i].set_ylabel(metric)
            ax[i].legend(["train", "val"])
        plt.show()

    # Testing
    model.load_weights("../output/3d_image_classification.h5")
    prediction = model.predict(X_test)

    # Calculating optimal threshold
    from sklearn import metrics as mt

    fpr, tpr, thresholds = mt.roc_curve(Y_test, prediction)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print('optimal threshold:', optimal_threshold)
    prediction = prediction > optimal_threshold

    # Calculating precision, recall and F1 score
    tn, fp, fn, tp = mt.confusion_matrix(Y_test, prediction).ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print("Precision: {} and Recall: {}".format(precision, recall))
    print("F1 score: {}".format(2 * precision * recall / (precision + recall)))
    import random

    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    fig.tight_layout()
    for i in range(4):
        c = random.randint(0, prediction.shape[0] - 1)
        axes[0, i].imshow(np.squeeze(X_test[c]))
        result = 'res'
        if prediction[c]:
            result = 'Positive'
        else:
            result = 'Negative'
        axes[0, i].set_title('Prediction: Corona {}'.format(result))
        axes[1, i].imshow(np.squeeze(X_test_inf[c]))
        axes[1, i].set_title('Actual mask')
    plt.show()
