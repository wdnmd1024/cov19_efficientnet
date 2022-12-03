import pathlib
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from tensorflow.keras import backend as K, losses
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Conv2DTranspose, concatenate
from tensorflow.keras.layers import UpSampling2D


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

    print(all_cts.shape)
    print(all_lungs.shape)
    print(all_inf.shape)

    # 数据同步混洗
    from sklearn.utils import shuffle
    all_cts, all_lungs, all_inf = shuffle(all_cts, all_lungs, all_inf)
    # 标准化
    all_cts = (all_cts - all_cts.min()) / (all_cts.max() - all_cts.min())
    all_lungs = (all_lungs - all_lungs.min()) / (all_lungs.max() - all_lungs.min())
    all_inf = (all_inf - all_inf.min()) / (all_inf.max() - all_inf.min())

    # 数据划分
    train_size = int(0.65 * all_cts.shape[0])
    X_train, yi_train = (all_cts[:train_size] / 255, all_inf[:train_size])
    X_valid, yi_valid = (all_cts[train_size:int(0.8 * all_cts.shape[0])] / 255,
                         all_inf[train_size:int(0.8 * all_cts.shape[0])])
    test_size = int(0.8 * all_cts.shape[0])
    X_test, yl_test, yi_test = (all_cts[test_size:]/255, all_lungs[test_size:], all_inf[test_size:])
    print(X_train.shape, yi_train.shape)
    print(X_valid.shape, yi_valid.shape)
    print(X_test.shape, yi_test.shape)

    return X_train, yi_train, X_valid, yi_valid, X_test, yi_test, yl_test


# 定义损失函数和准确度函数
def dice(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    dice = K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
    return dice


def dice_loss(y_true, y_pred):
    loss = 1 - dice(y_true, y_pred)
    return loss


def bce_dice_loss(y_true, y_pred):
    loss = 0.5 * losses.binary_crossentropy(y_true, y_pred) + 0.5 * dice_loss(y_true, y_pred)
    return loss


class CosineAnnealingLearningRateSchedule(callbacks.Callback):
    def __init__(self, n_epochs, n_cycles, lrate_max, verbose=0):
        self.epochs = n_epochs
        self.cycles = n_cycles
        self.lr_max = lrate_max
        self.lrates = list()

    def cosine_annealing(self, epoch, n_epochs, n_cycles, lrate_max):
        epochs_per_cycle = np.floor(n_epochs / n_cycles)
        cos_inner = (np.pi * (epoch % epochs_per_cycle)) / (epochs_per_cycle)
        return lrate_max / 2 * (np.cos(cos_inner) + 1)

    def on_epoch_begin(self, epoch, logs=None):
        lr = self.cosine_annealing(epoch, self.epochs, self.cycles, self.lr_max)
        K.set_value(self.model.optimizer.lr, lr)
        self.lrates.append(lr)


# U-net pro
def block1(input_shape, filtersize, poolsz=(2, 2)):
    x = Conv2D(filtersize, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal")(input_shape)
    x = Conv2D(filtersize, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal")(x)
    x_inter = BatchNormalization()(x)
    x = MaxPooling2D(poolsz)(x_inter)
    x = Dropout(0.2)(x)
    return x, x_inter


def block2(input_shape, filtersize):
    x = BatchNormalization()(input_shape)
    x = Conv2D(filtersize, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal")(x)
    x = Conv2D(filtersize, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal")(x)
    return x


def infection_segmentation():
    x_input = Input((128, 128, 1))

    x, Xa = block1(x_input, 32)
    x, Xb = block1(x, 64)
    x, _ = block1(x, 128, poolsz=(1, 1))
    x, _ = block1(x, 256, poolsz=(1, 1))
    x = block2(x, 256)

    x = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x)
    x = block2(x, 128)

    x = Conv2DTranspose(64, (2, 2), padding='same')(x)
    x = concatenate([x, Xb])
    x = block2(x, 64)

    x = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(x)
    x = concatenate([x, Xa], axis=3)
    x = block2(x, 32)

    infection_segmentation = Conv2D(1, (1, 1), activation='sigmoid', name='infect_output')(x)

    model = Model(inputs=x_input, outputs=infection_segmentation, name='infect_model')

    return model


if __name__ == '__main__':

    infection_segmentation = infection_segmentation()
    print(infection_segmentation.summary())

    # 加载数据集
    X_train, Y_train, X_val, Y_val, X_test, Y_test, yl_test = load_data()

    # 编译模型
    epochs = 100
    lrmax = 5e-5
    n_cycles = epochs / 25
    lr_cb = CosineAnnealingLearningRateSchedule(epochs, n_cycles, lrmax)
    checkpoint_fpath = "../output/infection_segmentation_weights.hdf5"
    cts_checkpoint_cb = callbacks.ModelCheckpoint(checkpoint_fpath,
                                                  monitor='val_dice',
                                                  save_best_only=True,
                                                  mode='max',
                                                  verbose=1,
                                                  save_weights_only=True)
    batch_size = 8
    optim = optimizers.Adam(lr=5e-5, beta_1=0.9, beta_2=0.99)
    infection_segmentation.compile(optimizer=optim, loss=bce_dice_loss, metrics=[dice])

    # Train
    path = pathlib.Path("../output/infection_segmentation_weights.hdf5")
    if not path.exists():
        infection_segmentation_res = infection_segmentation.fit(x=X_train,
                                                                y=Y_train,
                                                                batch_size=batch_size,
                                                                epochs=epochs,
                                                                verbose=1,
                                                                validation_data=(X_val, Y_val),
                                                                callbacks=[cts_checkpoint_cb, lr_cb])
        plt.style.use('ggplot')

        fig, axes = plt.subplots(1, 2, figsize=(11, 5))

        axes[0].plot(infection_segmentation_res.history['dice'], color='b', label='train-infection')
        axes[0].plot(infection_segmentation_res.history['val_dice'], color='r', label='valid-infection')
        axes[0].set_ylabel('Dice coefficient')
        axes[0].set_xlabel('Epoch')
        axes[0].legend()

        axes[1].plot(infection_segmentation_res.history['loss'], color='b', label='train-infection')
        axes[1].plot(infection_segmentation_res.history['val_loss'], color='r', label='valid-infection')
        axes[1].set_ylabel('Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].legend()
        plt.show()

    # Test
    infection_segmentation.load_weights('../output/infection_segmentation_weights.hdf5')
    prediction = infection_segmentation.predict(X_test)
    import random
    indices = random.choices(range(len(X_test)), k=5)
    fig, axes = plt.subplots(3, 5, figsize=(16, 10))
    for ii, idx in enumerate(indices):
        axes[0, ii].imshow(X_test[idx][:, :, 0], cmap='bone')
        axes[0, ii].set_title('CT image')
        plt.grid(None)
        axes[0, ii].set_xticks([])
        axes[0, ii].set_yticks([])

        axes[1, ii].imshow(yl_test[idx][:, :, 0], cmap='bone')
        axes[1, ii].imshow(Y_test[idx][:, :, 0], alpha=0.5, cmap='Reds')
        axes[1, ii].set_title('Infection mask')
        plt.grid(None)
        axes[1, ii].set_xticks([])
        axes[1, ii].set_yticks([])

        axes[2, ii].imshow(yl_test[idx][:, :, 0], cmap='bone')
        axes[2, ii].imshow(prediction[idx][:, :, 0], alpha=0.5, cmap='Reds')
        axes[2, ii].set_title('Pred. Infection mask')
        plt.grid(None)
        axes[2, ii].set_xticks([])
        axes[2, ii].set_yticks([])
    plt.show()

