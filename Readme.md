# DeViSE Milestone 3
### 檔案及其用處
* **config.cfg**: 參數設定檔，可以調整training/testing的參數
* **devise_model.py**: DeViSE model class, training/testing在這裡執行
* **visual_model.py**: 影像分類model class, training/testing在這裡執行
* **read_config.py**: 讀取設定檔
* **util.py**: 一些共用的function
* **main.py**: 主程式

### 使用前置作業
* 將cifar100 dataset解壓縮，並在設定檔內更改路徑
* 在設定檔內更改pre-train的word2vec model路徑

### 執行
python main.py