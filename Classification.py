import pickle

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization, Dense, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
import pathlib

from model import efficientnetv2_s as create_model

num_classes = 2

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


if __name__ == '__main__':
    # 加载模型
    model = get_model(width=128, height=128)

    #model = create_model(num_classes=num_classes)

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
