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
