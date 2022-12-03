import sys
import cv2 as cv
from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib.pyplot as plt
import numpy as np
from Cov19 import Classification
from Cov19 import Infection_Segmentation


class UI_Dialog(object):
    def setupUI(self, Dialog):
        Dialog.setObjectName('Dialog')
        Dialog.resize(590, 500)

        # 路径显示框
        self.path_edit = QtWidgets.QLineEdit(Dialog)
        self.path_edit.setGeometry(QtCore.QRect(10, 10, 251, 20))
        self.path_edit.setObjectName("path_edit")

        # 路径选择器
        self.toolButton = QtWidgets.QToolButton(Dialog)
        self.toolButton.setGeometry(QtCore.QRect(270, 10, 31, 21))  # 横坐标、纵坐标、宽、高
        self.toolButton.setObjectName("toolButton")

        # 开始诊断按钮
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(310, 10, 75, 23))
        self.pushButton.setObjectName("pushButton")

        # 原图
        self.original_ct = QtWidgets.QGraphicsView(Dialog)
        self.original_ct.setGeometry(QtCore.QRect(50, 60, 200, 200))
        self.original_ct.setObjectName("original_ct")
        self.original_label = QtWidgets.QLabel(Dialog)
        self.original_label.setGeometry(QtCore.QRect(50, 60, 200, 200))
        self.original_label.setObjectName("original_label")
        self.original_label.setScaledContents(True)  # label自适应图片大小

        # 增强之后的图片
        self.processed_ct = QtWidgets.QGraphicsView(Dialog)
        self.processed_ct.setGeometry(QtCore.QRect(300, 60, 200, 200))
        self.processed_ct.setObjectName("processed_ct")
        self.processed_label = QtWidgets.QLabel(Dialog)
        self.processed_label.setGeometry(QtCore.QRect(300, 60, 200, 200))
        self.processed_label.setObjectName("processed_label")
        self.processed_label.setScaledContents(True)  # label自适应图片大小

        # 灰度直方图
        self.hist_img = QtWidgets.QGraphicsView(Dialog)
        self.hist_img.setGeometry(QtCore.QRect(50, 280, 200, 200))
        self.hist_img.setObjectName("hist_img")
        self.hist_label = QtWidgets.QLabel(Dialog)
        self.hist_label.setGeometry(QtCore.QRect(50, 280, 200, 200))
        self.hist_label.setObjectName("hist_label")
        self.hist_label.setScaledContents(True)  # label自适应图片大小

        # 诊断之后的图片
        self.result_ct = QtWidgets.QGraphicsView(Dialog)
        self.result_ct.setGeometry(QtCore.QRect(300, 280, 200, 200))
        self.result_ct.setObjectName("result_ct")
        self.result_label = QtWidgets.QLabel(Dialog)
        self.result_label.setGeometry(QtCore.QRect(300, 280, 200, 200))
        self.result_label.setObjectName("result_label")
        self.result_label.setScaledContents(True)  # label自适应图片大小

        # 诊断结果
        self.label_1 = QtWidgets.QLabel(Dialog)
        self.label_1.setGeometry(QtCore.QRect(400, 10, 75, 23))
        self.label_1.setObjectName("label_1")
        self.label_1.setScaledContents(True)  # label自适应图片大小

        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(480, 10, 100, 23))
        self.label_2.setObjectName("label_2")
        self.label_2.setScaledContents(True)  # label自适应图片大小

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Covid2019 肺炎诊断系统"))
        self.path_edit.setText(_translate("Dialog", "E:/"))
        self.toolButton.setText(_translate("Dialog", "..."))
        self.pushButton.setText(_translate("Dialog", "开始诊断"))
        self.original_label.setText(_translate("Dialog", "待测CT图像"))
        self.processed_label.setText(_translate("Dialog", "处理增强之后的CT图像"))
        self.hist_label.setText(_translate("Dialog", "CT像素分布图"))
        self.result_label.setText(_translate("Dialog", "识别结果"))
        self.label_1.setText(_translate("Dialog", "诊断结果"))


class Myshow(QtWidgets.QWidget, UI_Dialog):
    def __init__(self):
        super(Myshow, self).__init__()
        self.setupUI(self)
        self.pushButton.clicked.connect(self.Recognition)
        self.toolButton.clicked.connect(self.ChoosePath)

        self.test_path = 'E:/'
        self.processed_path = 'E:/'
        self.pre_path = 'E:/'

    def ChoosePath(self):
        file_name = QtWidgets.QFileDialog.getOpenFileName(self, "open file dialog", self.test_path, "图片(*.png)")
        print(file_name[0])
        # 显示待测CT图
        self.test_path = file_name[0]
        self.path_edit.setText(self.test_path)
        self.original_label.setPixmap(QtGui.QPixmap(self.test_path))

        # 显示处理之后的CT图
        s = self.test_path.split('/')
        s[-2] = 'processed_cts'
        self.processed_path = '/'.join(s)
        self.processed_label.setPixmap(QtGui.QPixmap(self.processed_path))

        s[-2] = 'inf_mask'
        self.pre_path = '/'.join(s)

        self.label_2.clear()
        self.label_2.setStyleSheet("background-color:white")

        self.result_label.clear()
        self.result_label.setText(QtCore.QCoreApplication.translate("Dialog", "识别结果"))


        self.hist_label.setText(QtCore.QCoreApplication.translate("Dialog", "CT像素分布图"))
        # 显示像素直方图
        org_img = cv.imread(self.test_path, cv.IMREAD_GRAYSCALE)
        pro_img = cv.imread(self.processed_path, cv.IMREAD_GRAYSCALE)
        fig, axes = plt.subplots(1, 1, figsize=(5, 5))
        org_img, pro_img = np.uint8(org_img * 255), np.uint8(pro_img * 255)
        axes.hist(org_img.flatten(), alpha=0.3, color='skyblue', label='Original')
        axes.hist(pro_img.flatten(), alpha=0.3, color='red', label='Processed')
        axes.legend()
        # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        hist_path = '../output/hist_img.png'
        plt.savefig(hist_path, pad_inches=0.0)
        self.hist_label.setPixmap(QtGui.QPixmap(hist_path))

    def Recognition(self):
        cnn_model = Classification.get_model()
        processed_img = cv.imread(self.processed_path, cv.IMREAD_GRAYSCALE)
        img = np.reshape(processed_img, (1, 128, 128, 1))
        img = np.array(img)
        img = (img - img.min()) / (img.max() - img.min())
        prediction = cnn_model.predict(img)
        prediction = prediction > 0.4976595
        print(prediction)
        if prediction:
            self.label_2.setStyleSheet("color:blue")
            self.label_2.setStyleSheet("background-color:red")
            self.label_2.setText(QtCore.QCoreApplication.translate('Dialog', "新冠肺炎阳性"))
        else:
            self.label_2.setStyleSheet("color:blue")
            self.label_2.setStyleSheet("background-color:gold")
            self.label_2.setText(QtCore.QCoreApplication.translate('Dialog', "新冠肺炎阴性"))
        pre_inf = cv.imread(self.pre_path, cv.IMREAD_GRAYSCALE)
        pre_path_0 = '../output/pre_img.png'
        fig, axes = plt.subplots(1, 1, figsize=(15, 15))
        axes.imshow(processed_img, cmap='bone')
        axes.imshow(pre_inf, alpha=0.5, cmap='nipy_spectral')
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.savefig(pre_path_0, pad_inches=0.0)
        self.result_label.setPixmap(QtGui.QPixmap(pre_path_0))


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = Myshow()
    w.show()
    sys.exit(app.exec_())
