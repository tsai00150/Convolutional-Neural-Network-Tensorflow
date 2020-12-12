# Convolutional-Neural-Network-Tensorflow

小組組員：謝采辰、李朋泰、張家齊、溫祐宇\
以下內容之程式碼源自於Tensorflow官網的CNN教學範本，所有程式皆以此為基礎進行改寫，連結如下👇\
https://www.tensorflow.org/tutorials/images/cnn?hl=zh-tw


## 一、Tensorflow 與 Keras簡介

Tensrflow是一個由Google旗下的團隊「Google Brain Team」所開發的開放原始碼軟體資料庫，於2015年11月釋出初始版本，隨後被50個開發團隊採用，廣用於製作Google與相關的商業產品的機器學習工具，便取代了Google在2010年開始使用的第一代機器學習系統「DistBelief」的深度學習任務，成為很多我們日常生活中習以為常的服務功能，例如Gmail過濾垃圾信、Google翻譯、Google搜尋、Google語音搜尋、Google語音辨識、Google圖片辨識、Google相簿、Google地圖、Google街景、YouTube和網頁廣告投放......等等。\
TensorFlow取名靈感來自於對多維陣列執行的操作，這些多維陣列被稱為張量(Tensor)，透過運算張量模擬神經網路，而它主要就是要藉由實現高強度的矩陣運算去達到機器學習的最高效能，而它的另一個優點是讓不同平台、作業系統和語言的用戶都能不修改程式碼直接使用，它的底層核心引擎由C++實現，通過gRPC實現網路互訪、分散式執行，並且提供了一個Python API，以及C++、Haskell、Java、Go和Rust等等其他語言的 API，若使用第三方程式還可用於 C#、.NET Core、Julia、R和Scala。\
雖然它的Python、C++和Java的API共享了大部分執行代碼，但是有關於反向傳播梯度計算的部分仍然需要在不同語言單獨實現，而目前只有Python API較為豐富的實現了反向傳播部分，所以大多數人，包括我們的這次報告都是使用Python進行模型訓練，但是依然可以選擇使用其它語言進行線上推理。

![Alt text](readme-images/img0.jpg?raw=true "Title")

由於Tensorflow屬於比較低階的深度學習API，好處是相當靈活，可適應各種應用，搭配各種設計。缺點就是在設計模型時，像是張量乘積、卷積等等基礎操作統統都得使用者自己來處理，相當費時，而且初學者較難理解與學習。於是開發社群就以Tensorflow為基礎，進而開發高階深度學習API，讓用戶可以使用更簡潔的方式去建立模型，其中最完整的莫過於「Keras」。\
Keras是一個開放原始碼，高階的深度學習程式庫，使用Python編寫，旨在快速實現包含卷積神經網路在內的各種深度神經網路，並專注於用戶友好、模組化和可延伸性。其代碼放在GitHub上，包含許多常用神經網路構建塊的實現，例如層、目標、啟用功能、優化器和一系列工具，其他常見的實用公共層支援還有Dropout與池化層等等，讓使用者可以更輕鬆地處理圖像和文字資料。\
自2015年3月釋出第一版之後開始受到深度學習的學習者愛用，於2017年開始接受Google團隊的支援，得以用TensorFlow作為後端引擎，兩者搭配將使建立深度學習模型與進行訓練、預測更加快速和簡易。

![Alt text](readme-images/img1.jpg?raw=true "Title")

透過Keras所提供的各項模組工具，例如卷積和池化模組，我們得以較輕鬆地操作一個類似上圖結構的神經網路，並透過調整各個細節的改變去研究每個環節對訓練與預測會造成那些影響。

## 二、Tensorflow與VGG-16範例 介紹

### (1)Tensorflow 語法介紹
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

### (2)VGG-16介紹
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

圖：VGG-16示意圖
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

## 三、變數修改與結果
### (1)調整Epoch的數量
我們透過調整epoch的數量來觀察模型準確率的變化，以epoch數量5,10,15,20,25,30來做測試，每個數量均以程式各跑5次來求出平均：\
epoch = 5, accuracy = 0.6901\
epoch = 10, accuracy = 0.7143\
epoch = 15, accuracy = 0.7023\
epoch = 20, accuracy = 0.7001\
epoch = 25, accuracy = 0.6988\
epoch = 30, accuracy = 0.6885\
可以看出在數量達到15以上之後，準確率反而下滑。
後來我們針對epoch = 10做進一步測試，列出5,8,9,10,11,12,15個的測試。
平均值為：\
epoch = 5, accuracy = 0.6911\
epoch = 8, accuracy = 0.6975\
epoch = 9, accuracy = 0.6997\
epoch = 10, accuracy = 0.7115\
epoch = 11, accuracy = 0.7056\
epoch = 12, accuracy = 0.7090\
epoch = 15, accuracy = 0.7043\
依然是epoch = 10為最高點，因此我們覺得，epoch數量若大於10個，在此模型就會產生over fitting的現象，進而導致準確率下降。

### (2)Learning Rate
透過更改步距來觀察準確率的變化，為了更改步距，必須import optimizers：
```py
from keras import optimizers
# compile model時設定adam的learning rate, 我們分別測試了0.1, 0.01, 0.001, 0.0001四種
adam = optimizers.Adam(lr=learning rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
```
以下是測試結果：\
Lr = 0.1, Test Accuracy = 0.6871\
Lr = 0.01, Test Accuracy = 0.7164\
Lr = 0.001, Test Accuracy = 0.7060\
Lr = 0.0001, Test Accuracy = 0.6774\
從測試結果來看，對於這個模型來說，Learning rate = 0.01所得出的準確率是最高的，因此設為將步距設為0.01最為合適我們的data。

### (3)Data Preprocessing
完整程式碼：```Data Preprocessing.ipynb```\
將資料先經過預先處理，標準化資料之後，能夠優化我們的模型；
載入資料庫的圖片後，對圖片做標準化當作preprocessing：
```py
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
def standardize(input):
    mean = np.mean(input)
    std = np.std(input)
    return (input - mean) / std
train_images, test_images = standardize(train_images)  , standardize(test_images)
```
以下是測試結果：\
**loss: 0.7044 - acc: 0.7550 - val_loss: 0.8190 - val_acc: 0.7184**\
Test Accuracy = 0.7184\
相較之下，進行過data preprocessing的模型預測準確率，提高了大約4~5個百分點，效果相當顯著。

### (4)Weight Initialization
深度學習模型訓練的過程本身就是在調整weight的大小，但如何選擇參數初始值就是一個重要的環節，因為對於結構龐大的學習模型，非線性函數不斷疊加，因此選擇良好的初始值可大大增加模型的效率。\
我們選擇了兩種initial function來介紹：
#### i. Random initialization
在convolutional, dense layer做初始化：
```py
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), 
                       kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', 
                       kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', 
                       kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu', 
                      kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)))
model.add(layers.Dense(10, activation='softmax', 
                      kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)))
```
測試結果：\
**loss: 0.8697 - acc: 0.6945 - val_loss: 0.9320 - val_acc: 0.6710**\
Test Accuracy = 0.6710
#### ii. Truncated initialization
在convolutional, dense layer做初始化：
```py
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), 
                       kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', 
                       kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', 
                       kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu', 
                      kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)))
model.add(layers.Dense(10, activation='softmax', 
                      kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)))
```
測試結果：\
**loss: 0.8825 - acc: 0.6868 - val_loss: 0.9490 - val_acc: 0.6715**\
Test Accuracy = 0.6715\
由測試結果得知，這兩個weight initialization的效果對於我們的模型準確度皆沒有顯著的影響。

### (5)Batch Normalization
是將分散的數據統一標準化的一種做法，同樣的能夠優化我們的神經網路。將batch function加在maxpooling, dense layer之後：
```py
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='sigmoid', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, (3, 3), activation='sigmoid'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, (3, 3), activation='sigmoid'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='sigmoid'))
model.add(layers.BatchNormalization())
model.add(layers.Dense(10, activation='softmax'))
```
測試結果：
**loss: 0.7193 - acc: 0.7504 - val_loss: 0.9392 - val_acc: 0.6816**\
Test Accuracy = 0.6816\
由測試結果可看出，batch normalization的效果對於我們的模型並沒有顯著的影響。

### (6)Optimizer的調整
Optimizer對於一個學習模型來說相當重要，隨著技術演化，新開發出的Optimizer可以得到更高的準確度以及更快的速度。
#### i.使用Adadelta
```py
model.compile(optimizer='Adadelta', # 預設為lr=1.0, rho=0.95, epsilon=1e-06
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
```
Test Accuracy = 0.7339
#### ii.使用Adagrad
```py
model.compile(optimizer='Adagrad', # 預設為lr=0.01, epsilon=1e-06
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
```
Test Accuracy = 0.7436
#### iii.使用RMSprop
```py
model.compile(optimizer='RMSprop', # 預設為lr=0.001, rho=0.9, epsilon=1e-06
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
```
Test Accuracy = 0.6951
#### iiii.使用sgd
```py
model.compile(optimizer='sgd', # 預設為lr=0.01, momentum=0.0, decay=0.0, nesterov=False
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
```
Test Accuracy = 0.5637\
再加上我們原本使用的adam，我們共測試了五種的optimizer，可看出對於我們的資料，模型準確率最高的依序是Adagrad、Adadelta、RMSprop、adam、sgd，但這並不代表所有資料都適用於這個結論。

### (7)Dropout
當學習模型容量過高或者是訓練資料過少很容易造成over fitting的問題，這時候利用dropout以一定的機率丟棄隱藏層神經元，這樣在反向傳播時，被丟棄的神經元梯度是 0，所以在訓練時不會過度依賴某一些神經元，藉此達到預防over fitting的效果。
#### i.將dropout機率設為0.2
```py
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(10, activation='softmax'))
```
Test Accuracy = 0.6832
#### ii.將dropout機率設為0.3
```py
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(10, activation='softmax'))
```
Test Accuracy = 0.6586
#### iii.將dropout機率設為0.4
```py
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(10, activation='softmax'))
```
Test Accuracy = 0.6476
#### iiii.將dropout機率設為0.5
```py
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))
```
Test Accuracy = 0.6554\
從結果看出，除了dropout=0.22得到的準確率略高一點以外，其餘的皆差不到一個百分點，因此將dropout機率設為0.2應是最適合我們的資料。

### (8) 最佳模型
我們將個別參數最好的結果放在一起，發現準確率反而下降，大約在0.52左右，可見參數的設定必須視情況而定，並沒有標準答案。
因此，我們根據VGG16為樣本進行修改，將層數縮小，原先的VGG16有五個block，但因我們輸入之圖片大小為32x32，因此，我們刪掉至三個block。而其他參數保持不變，像是filter為3x3、兩層卷積一層池化、權重初始化用he_normal、relu等。\
此外，我們還採用data preprocessing、data augmentation、dropout等技術，以增加資料量並減少overfitting的問題。
完整程式碼在```OwnImageTest.ipynb```中；Test Accuracy = 0.7898

## 四、自行測試圖片
訓練完成之後，可以用自身的圖片來測試結果。
首先，在整個model皆完成訓練後，新增一個save來把用CIFAR-10訓練完成的model存起來：
```py
model.save('model.h5')
```
系統會將它儲存在和執行的.py檔同一個資料夾，若有重新訓練過，系統會將它覆寫。
下一步，則是將之前儲存的model 提出來：
```py
from keras.models import load_model
loaded_model = tf.keras.models.load_model('model.h5')
loaded_model.layers[0].input_shape
loaded_model.evaluate(test_images, test_labels)
```
最後，我們寫了兩個函式，分別是input和process。input為一簡單函式，將要輸入的圖片在電腦內的路徑用array的形式去儲存，再將它一一提出並且讀取、判斷。process則需要使用到OpenCV來讀取圖片並且轉為32x32x3的形式，藉由predict來把之前存取的模型套在圖片上。最後，顯示分數最高的class名稱。\
完整程式碼在```OwnImageTest.ipynb```中
```py
import cv2 as cv
def input(pictureArray):
    for i in range(0, len(pictureArray)):
        process(pictureArray[i])

def process (img):
    image = cv.imread(img, cv.IMREAD_COLOR)
    image = cv.resize(image, (32, 32))
    image = image.astype('float32')
    image = image.reshape(1, 32, 32, 3)
    image = 255-image
    image /= 255
    plt.show()
    pred = loaded_model.predict(image.reshape(1, 32, 32, 3), batch_size=1)
    print('Prediction:' + class_names [pred.argmax()])
    img = load_img(img, target_size=(32, 32))
    plt.figure()
    plt.imshow(img) 
    plt.show()

pictureArray = ['C:/Users/user/Test Images/ship1.jpg', 
                'C:/Users/user/Test Images/ship2.jpg', 
		.
		.
		.
	#輸入多組圖片的當案路徑
					]
input(pictureArray)
```
我們將八張船以及六張貓的照片放進照片，結果訓練效果不符合在CIFAR-10上的訓練準確度。我們認為原因來自於我們的訓練集CIFAR-10是32x32的圖，導致我們自己放進去的圖也要符合32x32的格式，在pixel數不夠高的情形下難以去辨識其好壞，因此錯誤率才會如此高。在未來，除了可以嘗試訓練影像資訊量更大的資料集外，也要再多嘗試不同的訓練參數，以獲得更好的訓練成績。

