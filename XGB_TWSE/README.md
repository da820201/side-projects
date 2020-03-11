# data_load.py
### 程式目的: 能夠將'批量'資料讀出，並分割成餵進MODEL前的理想樣式
### 功能介紹:
        1.detect_data_path: 此功能是做資料路徑的清理，並將list/str/glob.glob等type的格式轉換成統一的dict格式
        
        2.read_data: 此功能主要是從指定的csv中，選擇是否合併、刪除相同欄位
        
        3.train_test: 此功能能夠決定在資料進行維度轉換前，要丟棄那些指定的欄位、取出指定的欄位當作Y、幫助Y做Gaussian化(預設不做)、
                      幫助Y做One-hot-encode(預設是執行)，是否有指定One-hot-encode最大的值(莫認為列表總長)、是否只要X(不輸出)、是否只要
                      predict等等的功能
                      
        4.data_l: 此功能主要用於串起read_data以及train_test，是一種基於批量訓練而產生的自動化腳本
        
        5.auto_multilabels_to_one: 此功能主要是用於按照神經網路的設計，去按照相同比例批量生成多種input、output、loss、loss weight
                                   等等的參數
     
     
# XGB.py
### 程式目的: 批量訓練出能夠出個日漲跌趨勢的模型
### 功能介紹: 
        1.f1_error: 這是一種估量模型好壞的方式(F1-score)，同時也能幫助我們在訓練模型時，理解模型究竟是傾向'回答正確'還是'避免失誤'，
                    以此來得到修正模型的方向
        
        2.train_mode: 此功能主要用以指定特定股票、訓練集來進行訓練，這個功能主要也是串接的作用
        
        3.ceate_feature_map: 此功能主要是把XGB模型訓練(跌代)產生的樹、枝、葉等節點，依順序寫到指定的文件中
        
        4.draw_tree: 畫出決策樹的節點
        
        5.fig_fixing: 利用pyplot來改善draw_tree畫出決策樹可讀性低的問題
        
        6.save_model: 儲存模型，可以存成pickle、joblib、或只用xgb.save_model的方式儲存模型
        
        7.load_model: 讀取模型，如同儲存模型般，讀取也可以指定pickle或joblib的方式來讀取
        
        8.train_loop: 批量訓練模型，依照指定的股票代號，讀取所有預測目標(y)，透過串接上述的功能來進行所有股票的訓練，並批量儲存在指定的資料夾內


# XGB_for_load.py
### 程式目的: 
        1.批量產生預測數據
        
        2.批量繪製混淆矩陣
        
        3.批量批量透過LR產生分類數據
        
        4.批量批量寫入Lanten layer的預測結果(產生其他NN需求的數據)
        
        5.批量將結果推送至SQL
        
### 功能介紹: 
        1.XGB_loader: 這是個大功能，當初很急要產生結果，因此並沒有將功能拆分
        
                1.model_loader: 清洗模型路徑數據，返回一個字典，包含所有偵測到模型的路徑(凡副檔名為model)
                
                2.defult_xgb: 載入模型前，先預設一個xgb模型，以便加載擴展
                
                3.With_LR: 透過LR進行預測的版本，可以將LR或XGB輸出的節點寫成另一個數據集；也可以將XGB透過LR的預測結果批量推送到SQL；
                
                4.model_preds: 單純XGB的預測模型，讀取後可以選擇混淆矩陣、預測結果等兩種輸出
