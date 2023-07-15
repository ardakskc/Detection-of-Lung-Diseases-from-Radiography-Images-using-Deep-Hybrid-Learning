
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from keras.applications import ResNet50
from keras.applications.efficientnet import preprocess_input
from keras.models import *
import tensorflow as tf
import cv2
import numpy as np
import os
import pickle
import sklearn


class Ui_MainWindow(object):

    path = None
    base_model2 = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3),
        pooling='avg',
    )

    base_model2.load_weights('model_RN50_ss.h5', by_name=True)

    model_rn50 = Sequential([
        base_model2
    ])

    with open('svm.pkl', 'rb') as f:
        svm = pickle.load(f)


    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(920, 520)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setEnabled(True)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(530, 140, 361, 221))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_2 = QtWidgets.QLabel(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        self.tahmin = QtWidgets.QLabel(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.tahmin.setFont(font)
        self.tahmin.setObjectName("tahmin")
        self.horizontalLayout.addWidget(self.tahmin)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_4 = QtWidgets.QLabel(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_3.addWidget(self.label_4)
        self.sonuc = QtWidgets.QLabel(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.sonuc.setFont(font)
        self.sonuc.setObjectName("sonuc")
        self.horizontalLayout_3.addWidget(self.sonuc)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.resim = QtWidgets.QLabel(self.centralwidget)
        self.resim.setGeometry(QtCore.QRect(9, 9, 501, 501))
        self.resim.setText("")
        self.resim.setAlignment(QtCore.Qt.AlignCenter)
        self.resim.setObjectName("resim")
        self.test = QtWidgets.QPushButton(self.centralwidget)
        self.test.setGeometry(QtCore.QRect(530, 450, 361, 31))
        self.test.setObjectName("test")
        self.arama = QtWidgets.QPushButton(self.centralwidget)
        self.arama.setGeometry(QtCore.QRect(530, 30, 361, 31))
        self.arama.setObjectName("arama")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Lung Diseases Detector"))
        self.label_2.setText(_translate("MainWindow", "Tahmin:"))
        self.tahmin.setText(_translate("MainWindow", ""))
        self.label_4.setText(_translate("MainWindow", "Gerçek :"))
        self.sonuc.setText(_translate("MainWindow", ""))
        self.test.setText(_translate("MainWindow", "Test"))
        self.arama.setText(_translate("MainWindow", "Resim Seç"))
        self.arama.clicked.connect(self.tikla)
        self.test.clicked.connect(self.test_h)

    def test_h(self):
        classes = ['Covid-19', 'Normal', 'Opacity', 'Pneumonia']

        if self.path != None:

            feature, label = self.extract_features_and_labels()
            # Load the model from the file
            pred = self.svm.predict(feature)
            self.tahmin.setText(classes[pred[0]])

        else:
            self.sonuc.setText("Goruntu Seciniz.")

    def tikla(self):
        hmap = {'Viral Pneumonia': 'Pneumonia', 'Normal': 'Normal', 'COVID': 'Covid-19', 'Lung_Opacity': 'Opacity','':''}

        filename = QFileDialog.getOpenFileName()
        self.pixmap = QPixmap(filename[0])
        self.path = filename[0]
        label = os.path.basename(self.path).split("-")[0]

        self.tahmin.setText("")
        self.sonuc.setText(hmap[label])
        self.resim.setPixmap(self.pixmap)

    def extract_features_and_labels(self):

        imgSize = 224

        hmap = {'Viral Pneumonia': 'Pneumonia', 'Normal': 'Normal', 'COVID': 'Covid-19', 'Lung_Opacity': 'Opacity'}

        X = []
        Y = []

        label = os.path.basename(self.path).split("-")[0]
        #print(label)
        image = cv2.imread(self.path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (imgSize, imgSize))

        X.append(image)
        Y.append(hmap[label])

        X = np.array(X)
        Y = np.array(Y)

        features = []
        encoded_labels = []

        for image, label in zip(X, Y):
            # Görüntüyü ön işleme
            image = preprocess_input(image)
            image = np.expand_dims(image, axis=0)

            # Özellik vektörünü elde etme
            features_vector = self.model_rn50.predict(image)

            # Özellik ve etiketleri listelere ekleme
            features.append(features_vector.flatten())
            encoded_labels.append(label)

        return np.array(features), np.array(encoded_labels)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())