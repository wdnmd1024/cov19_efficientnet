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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers


# （1）swish激活函数
def swish(x):
    x = x * tf.nn.sigmoid(x)
    return x


# （2）标准卷积块
def conv_block(inputs, filters, kernel_size, stride, activation=True):
    # 卷积+BN+激活
    x = layers.Conv2D(filters=filters,
                      kernel_size=kernel_size,
                      strides=stride,
                      padding='same',
                      use_bias=False)(inputs)

    x = layers.BatchNormalization()(x)

    if activation:  # 如果activation==True就使用激活函数
        x = swish(x)

    return x


# （3）SE注意力机制
def se_block(inputs, in_channel, ratio=0.25):
    '''
    inputs: 深度卷积层的输出特征图
    input_channel: MBConv模块的输入特征图的通道数
    ratio: 第一个全连接层的通道数下降为MBConv输入特征图的几倍
    '''
    squeeze = int(in_channel * ratio)  # 第一个FC降低通道数个数
    excitation = inputs.shape[-1]  # 第二个FC上升通道数个数

    # 全局平均池化 [h,w,c]==>[None,c]
    x = layers.GlobalAveragePooling2D()(inputs)

    # [None,c]==>[1,1,c]
    x = layers.Reshape(target_shape=(1, 1, x.shape[-1]))(x)

    # [1,1,c]==>[1,1,c/4]
    x = layers.Conv2D(filters=squeeze,  # 通道数下降1/4
                      kernel_size=(1, 1),
                      strides=1,
                      padding='same')(x)

    x = swish(x)  # swish激活

    # [1,1,c/4]==>[1,1,c]
    x = layers.Conv2D(filters=excitation,  # 通道数上升至原来
                      kernel_size=(1, 1),
                      strides=1,
                      padding='same')(x)

    x = tf.nn.sigmoid(x)  # sigmoid激活，权重归一化

    # [h,w,c] * [1,1,c] ==> [h,w,c]
    outputs = layers.multiply([inputs, x])

    return outputs


# （3）逆转残差模块
def MBConv(x, expansion, kernel_size, stride, out_channel, dropout_rate):
    '''
    expansion: 第一个卷积层特征图通道数上升的倍数
    kernel_size: 深度卷积层的卷积核size
    stride: 深度卷积层的步长
    out_channel: 第二个卷积层下降的通道数
    dropout_rate: Dropout层随机丢弃输出层的概率，直接将输入接到输出
    '''
    # 残差边
    residual = x

    # 输入特征图的通道数
    in_channel = x.shape[-1]

    # ① 1*1标准卷积升维
    x = conv_block(inputs=x,
                   filters=in_channel * expansion,  # 上升通道数为expansion倍
                   kernel_size=(1, 1),
                   stride=1,
                   activation=True)

    # ② 3*3深度卷积
    x = layers.DepthwiseConv2D(kernel_size=kernel_size,
                               strides=stride,
                               padding='same',
                               use_bias=False)(x)

    x = layers.BatchNormalization()(x)

    x = swish(x)

    # ④ SE注意力机制，输入特征图x，和MBConv模块输入图像的通道数
    x = se_block(inputs=x, in_channel=in_channel)

    # ⑤ 1*1标准卷积降维，使用线性激活
    x = conv_block(inputs=x,
                   filters=out_channel,  # 上升通道数
                   kernel_size=(1, 1),
                   stride=1,
                   activation=False)  # 不使用swish激活

    # ⑥ 只有步长=1且输入等于输出shape，才使用残差连接输入和输出
    if stride == 1 and residual.shape == x.shape:

        # 判断是否进行dropout操作
        if dropout_rate > 0:
            # 参数noise_shape一定的概率将某一层的输出丢弃
            x = layers.Dropout(rate=dropout_rate,  # 丢弃概率
                               noise_shape=(None, 1, 1, 1))

        # 残差连接输入和输出
        x = layers.Add([residual, x])

        return x

    # 如果步长=2，直接输出1*1卷积降维后的结果
    return x


# （4）Fused-MBConv模块
def Fused_MBConv(x, expansion, kernel_size, stride, out_channel, dropout_rate):
    # 残差边
    residual = x

    # 输入特征图的通道数
    in_channel = x.shape[-1]

    # ① 如果通道扩展倍数expansion==1，就不需要升维
    if expansion != 1:
        # 3*3标准卷积升维
        x = conv_block(inputs=x,
                       filters=in_channel * expansion,  # 通道数上升为原来的expansion倍
                       kernel_size=kernel_size,
                       stride=stride)

    # ② 判断卷积的类型
    # 如果expansion==1，变成3*3卷积+BN+激活；
    # 如果expansion!=1，变成1*1卷积+BN，步长为1
    x = conv_block(inputs=x,
                   filters=out_channel,  # FusedMBConv模块输出特征图通道数
                   kernel_size=(1, 1) if expansion != 1 else kernel_size,
                   stride=1 if expansion != 1 else stride,
                   activation=False if expansion != 1 else True)

    # ④ 当步长=1且输入输出shape相同时残差连接
    if stride == 1 and residual.shape == x.shape:

        # 判断是否使用Dropout层
        if dropout_rate > 0:
            x = layers.Dropout(rate=dropout_rate,  # 随机丢弃输出层的概率
                               noise_shape=(None, 1, 1, 1))  # 代表不是杀死神经元，是丢弃输出层

        # 残差连接输入和输出
        outputs = layers.Add([residual, x])

        return outputs

    # 若步长等于2，直接输出卷积层输出结果
    return x


# （5）每个模块重复执行num次
# Fused_MBConv模块
def Fused_stage(x, num, expansion, kernel_size, stride, out_channel, dropout_rate):
    for _ in range(num):
        # 传入参数，反复调用Fused_MBConv模块
        x = Fused_MBConv(x, expansion, kernel_size, stride, out_channel, dropout_rate)

    return x


# MBConv模块
def stage(x, num, expansion, kernel_size, stride, out_channel, dropout_rate):
    for _ in range(num):
        # 反复执行MBConv模块
        x = MBConv(x, expansion, kernel_size, stride, out_channel, dropout_rate)

    return x


# （6）主干网络
def efficientnetv2(input_shape, classes, dropout_rate):
    # 构造输入层
    inputs = keras.Input(shape=input_shape)

    # 标准卷积层[224,224,3]==>[112,112,24]
    x = conv_block(inputs, filters=24, kernel_size=(3, 3), stride=2)

    # [112,112,24]==>[112,112,24]
    x = Fused_stage(x, num=2, expansion=1, kernel_size=(3, 3),
                    stride=1, out_channel=24, dropout_rate=dropout_rate)

    # [112,112,24]==>[56,56,48]
    x = Fused_stage(x, num=4, expansion=4, kernel_size=(3, 3),
                    stride=2, out_channel=48, dropout_rate=dropout_rate)

    # [56,56,48]==>[32,32,64]
    x = Fused_stage(x, num=4, expansion=4, kernel_size=(3, 3),
                    stride=2, out_channel=64, dropout_rate=dropout_rate)

    # [32,32,64]==>[16,16,128]
    x = stage(x, num=6, expansion=4, kernel_size=(3, 3),
              stride=2, out_channel=128, dropout_rate=dropout_rate)

    # [16,16,128]==>[16,16,160]
    x = stage(x, num=9, expansion=6, kernel_size=(3, 3),
              stride=1, out_channel=160, dropout_rate=dropout_rate)

    # [16,16,160]==>[8,8,256]
    x = stage(x, num=15, expansion=6, kernel_size=(3, 3),
              stride=2, out_channel=256, dropout_rate=dropout_rate)

    # [8,8,256]==>[8,8,1280]
    x = conv_block(x, filters=1280, kernel_size=(1, 1), stride=1)

    # [8,8,1280]==>[None,1280]
    x = layers.GlobalAveragePooling2D()(x)

    # dropout层随机杀死神经元
    if dropout_rate > 0:
        x = layers.Dropout(rate=dropout_rate)

        # [None,1280]==>[None,classes]
    logits = layers.Dense(classes)(x)

    # 构建网络
    model = Model(inputs, logits)

    return model






if __name__ == '__main__':
    # 加载模型
    #model = get_model(width=128, height=128)

    model = efficientnetv2(input_shape=[128, 128 1],  # 输入图像shape
                           classes=1000,  # 分类数
                           dropout_rate=0)
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
