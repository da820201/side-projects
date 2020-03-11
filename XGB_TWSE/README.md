# data_load.py
### 程式目的: 能夠將'批量'資料讀出並分割成餵進MODEL前的理想樣式
### 功能介紹:
        1.detect_data_path: 此功能是做資料路徑的清理，並將list/str/glob.glob等type的格式轉換成統一的dict格式
        
        2.read_data: 此功能主要是從指定的csv中，選擇是否合併、刪除相同欄位
        
        3.train_test: 此功能能夠決定在資料進行維度轉換前，要丟棄那些指定的欄位、取出指定的欄位當作Y、幫助Y做Gaussian化(預設不做)、
                      幫助Y做One-hot-encode(預設是執行)，是否有指定One-hot-encode最大的值(莫認為列表總長)、是否只要X(不輸出)、是否只要
                      predict等等的功能
# XGB.py
