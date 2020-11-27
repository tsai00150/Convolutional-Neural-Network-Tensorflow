# Convolutional-Neural-Network-Tensorflow

小組組員：謝采辰、李朋泰、張家齊、溫祐宇\
以下內容之程式碼源自於Tensorflow官網的CNN教學範本，所有程式皆以此為基礎進行改寫，連結如下👇\
https://www.tensorflow.org/tutorials/images/cnn?hl=zh-tw


## 一、Tensorflow 與 Keras簡介

Tenserflow是一個由Google旗下的團隊「Google Brain Team」所開發的開放原始碼軟體資料庫，於2015年11月釋出初始版本，隨後被50個開發團隊採用，廣用於製作Google與相關的商業產品的機器學習工具，便取代了Google在2010年開始使用的第一代機器學習系統「DistBelief」的深度學習任務，成為很多我們日常生活中習以為常的服務功能，例如Gmail過濾垃圾信、Google翻譯、Google搜尋、Google語音搜尋、Google語音辨識、Google圖片辨識、Google相簿、Google地圖、Google街景、YouTube和網頁廣告投放......等等。\
TensorFlow取名靈感來自於對多維陣列執行的操作，這些多維陣列被稱為張量(Tensor)，透過運算張量模擬神經網路，而它主要就是要藉由實現高強度的矩陣運算去達到機器學習的最高效能，而它的另一個優點是讓不同平台、作業系統和語言的用戶都能不修改程式碼直接使用，它的底層核心引擎由C++實現，通過gRPC實現網路互訪、分散式執行，並且提供了一個Python API，以及C++、Haskell、Java、Go和Rust等等其他語言的 API，若使用第三方程式還可用於 C#、.NET Core、Julia、R和Scala。\
雖然它的Python、C++和Java的API共享了大部分執行代碼，但是有關於反向傳播梯度計算的部分仍然需要在不同語言單獨實現，而目前只有Python API較為豐富的實現了反向傳播部分，所以大多數人，包括我們的這次報告都是使用Python進行模型訓練，但是依然可以選擇使用其它語言進行線上推理。

![Alt text](readme-images/img0.jpg?raw=true "Title")

由於Tensorflow屬於比較低階的深度學習API，好處是相當靈活，可適應各種應用，搭配各種設計。缺點就是在設計模型時，像是張量乘積、卷積等等基礎操作統統都得使用者自己來處理，相當費時，而且初學者較難理解與學習。於是開發社群就以Tensorflow為基礎，進而開發高階深度學習API，讓用戶可以使用更簡潔的方式去建立模型，其中最完整的莫過於「Keras」。\
Keras是一個開放原始碼，高階的深度學習程式庫，使用Python編寫，旨在快速實現包含卷積神經網路在內的各種深度神經網路，並專注於用戶友好、模組化和可延伸性。其代碼放在GitHub上，包含許多常用神經網路構建塊的實現，例如層、目標、啟用功能、優化器和一系列工具，其他常見的實用公共層支援還有Dropout與池化層等等，讓使用者可以更輕鬆地處理圖像和文字資料。\
自2015年3月釋出第一版之後開始受到深度學習的學習者愛用，於2017年開始接受Google團隊的支援，得以用TensorFlow作為後端引擎，兩者搭配將使建立深度學習模型與進行訓練、預測更加快速和簡易。

![Alt text](readme-images/img1.jpg?raw=true "Title")

透過Keras所提供的各項模組工具，例如卷積和池化模組，我們得以較輕鬆地操作一個類似上圖結構的神經網路，並透過調整各個細節的改變去研究每個環節對訓練與預測會造成那些影響。

## Tensorflow與VGG-16範例 介紹

###### (1)Tensorflow 語法介紹
以下是我們報告之中所運行的基本程式碼：
```py
#第一步先導入Tensorflow、Keras和Matplotlib
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, initializers
import matplotlib.pyplot as plt
import random

#接著把圖片從資料庫中抓下來並分成十個類別
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

#再來是建立運算模型，經過三次卷積與兩次池化，簡化運算效率
#用relu函數使負值改為0，避免陰影，突現圖像的輪廓
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

#透過平坦層(Flatten Layer)將輸出壓成一維圖像
#回歸至全連階層(Dense Layer)進行分類
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()

#使用adam此一優化器(Optimizer)，編譯和訓練此模型，並建立起圖像結構
#有50000張圖片用於訓練，10000張圖片用於測試
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))
print(model.metrics_names)

#最後，繪製出展示訓練準確度和測試準確度的圖表
plt.plot(history.history['acc'], label='acc')
plt.plot(history.history['val_acc'], label = 'val_acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_acc)

#繪製結果範例如下
```
![Alt text](readme-images/img2.png?raw=true "Title")

###### (2)VGG-16介紹
下列為VGG-16套用於Tensorflow下的組成程式碼：
```
model = models.Sequential()

model.add(layers.Conv2D(64, (3, 3), activation='relu', 
input_shape=(32, 32, 3), padding="same"))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding="same"))
model.add(layers.MaxPooling2D((2, 2), strides=(2,2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu', padding="same"))
model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
model.add(layers.MaxPooling2D((2, 2), strides=(2,2)))

model.add(layers.Conv2D(256, (3, 3), activation='relu', padding="same"))
model.add(layers.Conv2D(256, (3, 3), activation='relu', padding="same"))
model.add(layers.Conv2D(256, (3, 3), activation='relu', padding="same"))
model.add(layers.MaxPooling2D((2, 2), strides=(2,2)))

model.add(layers.Conv2D(512, (3, 3), activation="relu", padding="same"))
model.add(layers.Conv2D(512, (3, 3), activation="relu", padding="same"))
model.add(layers.Conv2D(512, (3, 3), activation="relu", padding="same"))
model.add(layers.MaxPooling2D((2, 2), strides=(2,2)))

model.add(layers.Conv2D(512, (3, 3), activation="relu", padding="same"))
model.add(layers.Conv2D(512, (3, 3), activation="relu", padding="same"))
model.add(layers.Conv2D(512, (3, 3), activation="relu", padding="same"))
model.add(layers.MaxPooling2D((2, 2), strides=(2,2)))
```
來源：https://ithelp.ithome.com.tw/articles/10192162

圖：VGG-16示意圖\
![Alt text](readme-images/img3.jpg?raw=true "Title")\
訓練結果：
**loss: 2.3028 - acc: 0.0983 - val_loss: 2.3027 - val_acc: 0.1000**\
將Epoch設定為6之後，所得結果卻不如預期，訓練準確度一直在0.0973與0.0991之間遊走，至於測試準確度則是很固定維持在0.1。在一般的電腦上去做訓練的時間也相當長，平均每訓練完一次就要占用30分鐘的時間。\

為了了解這個現象，我們將VGG-16的層數減少，並且再跑一次。更改的項目以及結果如下：
```py
model = models.Sequential()

model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3), padding="same"))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding="same"))
model.add(layers.MaxPooling2D((2, 2), strides=(2,2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu', padding="same"))
model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
model.add(layers.MaxPooling2D((2, 2), strides=(2,2)))

model.add(layers.Conv2D(256, (3, 3), activation='relu', padding="same"))
model.add(layers.Conv2D(256, (3, 3), activation='relu', padding="same"))
model.add(layers.Conv2D(256, (3, 3), activation='relu', padding="same"))
model.add(layers.MaxPooling2D((2, 2), strides=(2,2)))
# 減少六個Conv2D層、兩個MaxPooling層、共八層
model.add(layers.Flatten())
model.add(layers.Dense(4096, activation='relu'))
#減少一個Dense層
model.add(layers.Dense(10, activation='softmax'))
```
訓練結果：
**loss: 0.6019 - acc: 0.7874 - val_loss: 0.7636 - val_acc: 0.7388**\
不難發現，訓練結果比起把所有的VGG-16都套用進CIFAR-10資料庫內好很多，甚至比原本官網所提供將近70%的準確度高出些許；至於訓練的速度，也是由層數較少的版本勝出。至於為何層數較少的版本反而執行起來的準確度高出許多?經過我們組內討論，這原因可以從Stanford的CNN教材中獲得解答。在第九章中的第81張投影片提到，若是增加太多層神經網路，在資料量不是特別高的訓練庫中的情形下，反而會增加錯誤率。CIFAR-10中的每一張圖皆是32x32的pixel數，但是VGG-16當初設計卻是輸入224x224的影像圖，我們認為訊息量上的落差會導致CIFAR-10的資料被過度解析，過多的層數也導致神經網路無法好好訓練。值得一提的是，此情況並非overfiiting，因為準確度並沒有因為訓練次數過多而降低--它本來就一直處於極低的情況。
圖：Stanford的CNN教材
![Alt text](readme-images/img4.jpg?raw=true "Title")

