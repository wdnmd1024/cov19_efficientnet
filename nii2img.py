import nibabel as nib
import tqdm.notebook as tqdm
import numpy as np
import cv2 as cv
import pandas as pd
import matplotlib.pyplot as plt

from Cov19 import Classification
from Cov19 import Infection_Segmentation

metadata = pd.read_csv('../input/metadata.csv')
clahe = cv.createCLAHE(clipLimit=7.0)  # 设置对比度阈值为3.0
img_size = 128


def data_transfer():
    original_cts = []
    all_cts = []
    all_inf = []
    bad_id = []
    for i in tqdm.tqdm(range(20)):
        cts = nib.load(metadata.loc[i, 'ct_scan'])
        lung = nib.load(metadata.loc[i, 'lung_mask'])
        inf = nib.load(metadata.loc[i, 'infection_mask'])

        a_cts = cts.get_fdata()
        a_lung = lung.get_fdata()
        a_inf = inf.get_fdata()

        slice_range = range(round(a_cts.shape[2] * 0.5), round(a_cts.shape[2] * 0.8), 15)

        a_cts = np.rot90(np.array(a_cts))
        a_cts = a_cts[:, :, slice_range]
        a_cts = np.reshape(np.rollaxis(a_cts, 2), (a_cts.shape[2], a_cts.shape[0], a_cts.shape[1], 1))

        a_lung = np.rot90(np.array(a_lung))
        a_lung = a_lung[:, :, slice_range]
        a_lung = np.reshape(np.rollaxis(a_lung, 2), (a_lung.shape[2], a_lung.shape[0], a_lung.shape[1], 1))

        a_inf = np.rot90(np.array(a_inf))
        a_inf = a_inf[:, :, slice_range]
        a_inf = np.reshape(np.rollaxis(a_inf, 2), (a_inf.shape[2], a_inf.shape[0], a_inf.shape[1], 1))

        # print(a_cts.shape)
        # print(a_lung.shape)
        # print(a_inf.shape)

        for j in range(a_cts.shape[0]):
            try:
                img_cts = cv.resize(a_cts[j], dsize=(img_size, img_size), interpolation=cv.INTER_AREA)
                img_lung = cv.resize(a_lung[j], dsize=(img_size, img_size), interpolation=cv.INTER_AREA)
                img_inf = cv.resize(a_inf[j], dsize=(img_size, img_size), interpolation=cv.INTER_AREA)

                xmax, xmin = img_cts.max(), img_cts.min()
                img_cts = (img_cts - xmin) / (xmax - xmin)
                original_ct = img_cts
                bounds = lung_mask_crop(img_lung)
                img_cts = clahe_enhancer(img_cts, [])
                img_cts = crop_lung(img_cts, bounds)
                all_cts.append(img_cts)
                all_inf.append(img_inf)
                original_cts.append(original_ct)
            except:
                bad_id.append(j)
    del_lst = []
    for i in tqdm.tqdm(range(len(all_cts))):
        try:
            all_cts[i] = cv.resize(all_cts[i], dsize=(img_size, img_size), interpolation=cv.INTER_AREA)
            all_cts[i] = np.reshape(all_cts[i], (img_size, img_size))
        except:
            del_lst.append(i)

        for idx in del_lst[::-1]:
            del all_cts[idx]
            del original_cts[idx]
            del all_inf[idx]
    return original_cts, all_cts, all_inf


def lung_mask_crop(mask):
    ht, wd = mask.shape
    _, thresh = cv.threshold(mask.astype('uint8'), 0.5, 1, 0)
    # 检测轮廓, 所有轮廓，仅保留拐点
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) < 2:
        raise Exception("Error")
    x0, y0, w0, h0 = cv.boundingRect(contours[0])
    x1, y1, w1, h1 = cv.boundingRect(contours[1])
    B = [min(x0, x1) - round(0.05 * wd), min(y0, y1) - round(0.05 * ht),
         max(x0 + w0, x1 + w1) - min(x0, x1) + round(0.1 * wd),
         max(y0 + h0, y1 + h1) - min(y0, y1) + round(0.1 * ht)]
    B = [max(B[0], 0), max(B[1], 0), min(B[2], wd), min(B[3], ht)]
    return B


def clahe_enhancer(img, axes):
    img = np.uint8(img * 255)
    clahe_img = clahe.apply(img)
    if len(axes) > 0:
        axes[0].imshow(img, cmap='bone')
        axes[0].set_title("Original")
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        axes[1].imshow(clahe_img, cmap='bone')
        axes[1].set_title("CLAHE")
        axes[1].set_xticks([])
        axes[1].set_yticks([])

    # 显示 CLAHE 增强图像与原始 CT 扫描图像的直方图
    if len(axes) > 2:
        axes[2].hist(img.flatten(), alpha=0.3, color='skyblue', label='Original')
        axes[2].hist(clahe_img.flatten(), alpha=0.3, color='red', label="CLAHE")
        axes[2].legend()
    return clahe_img


def crop_lung(img, boundaries):
    minx, miny, maxx, maxy = boundaries
    return img[miny:miny + maxy, minx:minx + maxx]


if __name__ == '__main__':
    test_path = "E:\\book\\2019151067_jht_task3\\test_cts\\"
    processed_path = "E:\\book\\2019151067_jht_task3\\processed_cts\\"
    original_cts, all_cts, all_inf = data_transfer()
    print(np.array(original_cts).shape, np.array(all_cts).shape, np.array(all_inf).shape)
    length = np.array(original_cts).shape[0]
    for i in range(length):
        plt.imsave("E:\\book\\2019151067_jht_task3\\test_cts\\" + str(i) + ".png", original_cts[i], cmap='bone')
        plt.imsave("E:\\book\\2019151067_jht_task3\\processed_cts\\" + str(i) + ".png", all_cts[i], cmap='bone')
        plt.imsave("E:\\book\\2019151067_jht_task3\\inf_mask\\" + str(i) + ".png", all_inf[i], cmap='bone')

    # all_cts = np.array(all_cts)
    # np.reshape(all_cts, (all_cts.shape[0], img_size, img_size, 1))
    # all_inf = np.array(all_inf)
    # all_cts = (all_cts - all_cts.min()) / (all_cts.max() - all_cts.min())
    # all_inf = (all_inf - all_inf.min()) / (all_inf.max() - all_inf.min())
    # Y_test = np.ones((len(all_inf)))
    # for i in range(len(all_inf)):
    #     if np.unique(all_inf[i]).size == 1:
    #         Y_test[i] = 0
    # model = Classification.get_model(width=128, height=128)
    # prediction = model.predict(all_cts)
    # # Calculating optimal threshold
    # from sklearn import metrics as mt
    #
    # fpr, tpr, thresholds = mt.roc_curve(Y_test, prediction)
    # optimal_idx = np.argmax(tpr - fpr)
    # optimal_threshold = thresholds[optimal_idx]
    # print('optimal threshold:', optimal_threshold)

