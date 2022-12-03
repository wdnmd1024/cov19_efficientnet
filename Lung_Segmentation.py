import pathlib

import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_XLA_FLAGS'] = '–tf_xla_enable_xla_devices'


def load_data():
    """=============== 加载数据 ==============="""
    infile = open('../input/Processed_Data.cp', 'rb')
    data_dict = pickle.load(infile)
    all_cts = data_dict['cts']
    all_inf = data_dict['infects']
    all_lungs = data_dict['lungs']
    infile.close()

    # 删除没有感染的样本
    index_arr = []
    inf_check = np.ones((1, len(all_inf)))
    for i in range(len(all_inf)):
        if np.unique(all_inf[i]).size == 1:
            inf_check[0, i] = 0
            index_arr.append(i)
    for i in index_arr[::-1]:
        all_cts = np.delete(all_cts, i, axis=0)
        all_inf = np.delete(all_inf, i, axis=0)
        all_lungs = np.delete(all_lungs, i, axis=0)

    # 数据同步混洗
    from sklearn.utils import shuffle
    all_cts, all_lungs = shuffle(all_cts, all_lungs)
    print(all_cts.shape)
    print(all_lungs.shape)

    # 数据归一化
    all_cts = (all_cts - all_cts.min()) / (all_cts.max() - all_cts.min())
    all_lungs = (all_lungs - all_lungs.min()) / (all_lungs.max() - all_lungs.min())

    """=============== 划分数据集6:2:2 ==============="""
    X_train = all_cts[:int(len(all_cts) * 0.6)]
    Y_train = all_lungs[:int(len(all_lungs) * 0.6)]
    X_val = all_cts[int(len(all_cts) * 0.6):int(len(all_cts) * 0.8)]
    Y_val = all_lungs[int(len(all_lungs) * 0.6):int(len(all_lungs) * 0.8)]
    X_test = all_cts[int(len(all_cts) * 0.8):]
    Y_test = all_lungs[int(len(all_lungs) * 0.8):]

    print("{} {}".format(X_train.shape, Y_train.shape))
    print("{} {}".format(X_val.shape, Y_val.shape))
    print("{} {}".format(X_test.shape, Y_test.shape))

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


# 定义损失函数和准确度函数
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


# 定义 U-Net 模型
def build_model():
    num_filters = [16, 32, 128, 256]
    inputs = Input((128, 128, 1))
    x = Conv2D(num_filters[0], kernel_size=3, activation='relu', padding='same')(inputs)
    x = MaxPooling2D(pool_size=2, padding='same')(x)

    x = Conv2D(num_filters[1], kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=2, padding='same')(x)

    x = Conv2D(num_filters[2], kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=2, padding='same')(x)

    x = Conv2D(num_filters[3], kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=2, padding='same')(x)

    x = Dense(num_filters[3], activation='relu')(x)

    x = UpSampling2D(size=2)(x)
    x = Conv2D(num_filters[3], kernel_size=3, activation='sigmoid', padding='same')(x)

    x = UpSampling2D(size=2)(x)
    x = Conv2D(num_filters[2], kernel_size=3, activation='sigmoid', padding='same')(x)

    x = UpSampling2D(size=2)(x)
    x = Conv2D(num_filters[1], kernel_size=3, activation='sigmoid', padding='same')(x)

    x = UpSampling2D(size=2)(x)
    lung_seg = Conv2D(1, kernel_size=3, activation='sigmoid', padding='same')(x)   # identifying lungs

    model = Model(inputs=inputs, outputs=lung_seg, name='lung_seg')

    return model


if __name__ == '__main__':
    model = build_model()
    print(model.summary())

    # 编译模型
    initial_learning_rate = 0.0001
    lr_schedule = optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
    )
    model.compile(
        loss=dice_coef_loss,
        optimizer=optimizers.Adam(learning_rate=lr_schedule),
        metrics=[dice_coef],
    )
    checkpoint_cb = callbacks.ModelCheckpoint(
        "../output/3d_image_segmentation.h5", save_best_only=True
    )

    # 加载数据集
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_data()

    # Train
    path = pathlib.Path("../output/3d_image_classification.h5")
    if not path.exists():
        history = model.fit(X_train, Y_train, epochs=120, validation_data=(X_val, Y_val), callbacks=[checkpoint_cb],  shuffle=True, verbose=1)

        plt.plot(history.history['dice_coef'])
        plt.plot(history.history['val_dice_coef'])
        plt.title('Model dice coeff')
        plt.ylabel('Dice coeff')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')
        plt.show()

    # Test
    model.load_weights('../output/3d_image_segmentation.h5')
    prediction = model.predict(X_test)
    import random

    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    fig.tight_layout()
    for i in range(4):
        c = random.randint(0, prediction.shape[0] - 1)
        axes[0, i].imshow(np.squeeze(prediction[c]))
        axes[0, i].set_title('Predicted')
        axes[1, i].imshow(np.squeeze(Y_test[c]))
        axes[1, i].set_title('Actual')
        axes[2, i].imshow(np.squeeze(X_test[c]))
        axes[2, i].set_title('CT Scan')
    plt.show()
