## CTR Prediction
本專案使用DeepFM模型來進行點擊率預測，並透過Optuna超參數調優與KFold交叉驗證，進一步提升模型的最佳性能和泛化能力

## Data Source
資料來源：Kaggle: Display Advertising Challenge
https://www.kaggle.com/competitions/criteo-display-ad-challenge

資料說明：
    - 數據欄位：40欄，數值型欄位共13欄；類別型欄位共有26欄，且經過哈希轉換
    - 數據筆數：4584萬
    - 標籤定義：0 = 未點擊；1 = 點擊

## Overview
目標：預測廣告點擊率
模型：DeepFM
調參方式：Optuna
交叉驗證：KFold
評估指標：
    (1).官方指定指標：Log Loss
    (2).AUC、ROC
    (3).Precision、Accuracy、Recall、F1-score

## EDA and Feature Engineering
1.資料前處理：
    (1).欄位整理：欄位命名、去除重複性資料
    (2).缺失值處理：
        - 移除缺失值超過百分之50的欄位
        - 類別型欄位：移除缺失值資料，並進行Label Encoder
        - 數值型欄位：使用Median填補缺失值
    (3).數據分佈調整：
        - 類別型欄位：保留資料語意
        - 數值型欄位：Boxcox轉換

2.特徵欄位篩選：
    (1).相關係數和欄位百分比
    (2).隨機森林特徵篩選
    (3).提升欄位百分比
    (4).使用PCA、移除異常值

## Final Result
Optuna最佳參數：Log Loss：0.4575/AUC：0.7840/Precision：0.6416/Accuracy：0.7882/Recall：0.3349/F1-score：0.4401

手動調整最佳參數：Log Loss：0.4546/AUC：0.7875/Precision：0.6525/Accuracy：0.7897/Recall：0.3295/F1-score：0.4379