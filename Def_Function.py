import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import skew, boxcox
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE


def RenameandCheckDuplicateandNaN(DataFrame):
    """
    1. 先將欄位名稱統一修改成 '預測目標' 和1, 2, 3依序直到欄位結束 >> 共有40欄位，故最後欄位名稱為39
    """
    DataFrame.columns = ['Predicted']+[str(i) for i in range(1, len(DataFrame.columns))]
    print(f'Rename is Finished.\n')

    """
    2. 檢查每個Chunk中是否有重複性資料，若有則刪除重複性資料，並印出計算共有幾筆
    """
    if DataFrame.duplicated().any():
        print('Duplicates Found.')
        Duplicated_Number = DataFrame.duplicated().sum()
        print(f'Duplicates Rows Count: {Duplicated_Number}')
        DataFrame.drop_duplicates(inplace=True)
        print(f'New Shape After Removing Duplicates: {DataFrame.shape}')
    else:
        print('No Duplicates Found.')
        print(f'DataFrame Shape: {DataFrame.shape}')

    """
    3. 計算每個Chunk中是否有缺值，並計算共有幾筆缺值
    """
    if DataFrame.isna().sum().any():
        print('NaN Found.')
        NaN_Number = DataFrame.isna().sum().sum()
        print(f'Total NaN Value: {NaN_Number}\n')
    else:
        print(f'No NaN Found.\n')

    return DataFrame

def ColumnsNaNSorted(DataFrame):
    """
    計算缺失的比例，並且依照比例排序，沒有缺失值欄位則排除不顯示
    """
    nan_percentage = (DataFrame.isna().sum() / DataFrame.shape[0]*100).sort_values(ascending=False)
    filtered_nan = nan_percentage[nan_percentage != 0]
    print(f'{filtered_nan}\n')

    return DataFrame

def RemovedColumnOver50(DataFrame):
    """
    移除缺失值大於50%的資料欄位，並且列出移除欄位名稱
    """
    nan_percentage = (DataFrame.isna().sum() / DataFrame.shape[0]*100).sort_values(ascending=False)
    columns_drop = nan_percentage[nan_percentage>50].index

    DataFrame.drop(columns=columns_drop, axis=1, inplace=True)

    print(f'Removed Columns List: {list(columns_drop)}\n')

    return DataFrame

def SeparateNumericalandObjective(DataFrame):
    """
    依照欄位類別分割資料，後續分開進行數據前處理
    """
    numerical_col = DataFrame.select_dtypes(include='number')
    objective_col = pd.concat([DataFrame['Predicted'], DataFrame.select_dtypes(include='object')], axis=1)

    return numerical_col, objective_col

def ColumnsHist(DataFrame):
    """
    除預測目標欄位繪製圓餅圖，查看資料是否平衡；其餘欄位繪製hist圖，觀察每個欄位的分佈
    """
    plt.figure(figsize=(25, 40))

    for i, col in enumerate(DataFrame):

        plt.subplot(10, 4, i+1)

        if col == 'Predicted':
            col_value_count = DataFrame[col].value_counts()
            plt.pie(col_value_count, labels=col_value_count.index, autopct='%1.1f%%')

            plt.title(f'Proporiton of {col}')
        else:
            cleaned_data = DataFrame[col].dropna()

            sns.histplot(cleaned_data, bins=30, kde=True, color='blue')

            plt.title(f'Distribution of Columns{col}')
            plt.xlabel('Values')
            plt.ylabel('Amount')

    plt.tight_layout()
    plt.show()

def WithoutLabelColumnsHist(DataFrame):
    """
    除預測目標欄位繪製圓餅圖，查看資料是否平衡；其餘欄位繪製hist圖，觀察每個欄位的分佈
    """
    plt.figure(figsize=(25, 40))

    for i, col in enumerate(DataFrame):

        plt.subplot(10, 4, i+1)

        if col != 'Predicted':
            sns.histplot(DataFrame[col], bins=30, kde=True, color='blue')

            plt.title(f'Distribution of Columns{col}')
            plt.xlabel('Values')
            plt.ylabel('Amount')

    plt.tight_layout()
    plt.show()

def CorrHeatMap(DataFrame):
    corr_col = DataFrame.corr()

    plt.figure(figsize=(10, 10))
    sns.heatmap(corr_col, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.show()
    
def CountStatistic(DataFrame):
    """
    分別計算每個欄位的眾數、中位數、平均值
    """
    for col in DataFrame:

        cleaned_col = DataFrame[col].dropna()

        col_mode = cleaned_col.mode().iloc[0]
        col_median = cleaned_col.median()    
        col_avg = round(cleaned_col.mean(), 3)

        print(f'Statistic of Columns{col}')
        print(f'Mode: {col_mode}')
        print(f'Median: {col_median}')
        print(f'AVG: {col_avg}\n')

def ImputeNaN(DataFrame):
    """
    填補缺失值，由於資料分佈皆為左偏分佈，若平均值大於中位數則填補中位數，保留分佈型態；反之，填補眾數維持資料型態
    """
    imputer_mode = SimpleImputer(strategy='most_frequent')
    imputer_median = SimpleImputer(strategy='median')

    for col in DataFrame.columns:
        if DataFrame[col].mean() > DataFrame[col].median():
            DataFrame[[col]] = imputer_median.fit_transform(DataFrame[[col]])
        else:
            DataFrame[[col]] = imputer_mode.fit_transform(DataFrame[[col]])
    
    print(f'{DataFrame.isna().sum()}\n')

    return DataFrame

def ObjectiveLabelEncoder(DataFrame, save_path='encoder_dict.joblib'):
    """
    原本資料已經被Hashed成32位元資料，除缺失值外，其他轉換成Label
    當dict為空和檔案存在，則讀取dict檔案沿用先前Chunk的數據轉換，保證每個Chunk數值保持一致
    若dict為空，則創建新的字典
    """
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        encoder_dict = joblib.load(save_path)
        print('Success Load encoder_dict')
    else:
        print('encoder_dict is None')
        encoder_dict = {}

    for col in DataFrame.columns:

        non_missing = DataFrame[col].dropna()

        if col in encoder_dict:
            encoder = encoder_dict[col]
        else:
            encoder = LabelEncoder()
            encoder.fit(non_missing)
            encoder_dict[col] = encoder
        
        unknown_label = set(non_missing) - set(encoder.classes_)
        if unknown_label:

            encoder_classes = list(map(str, encoder.classes_))
            unknown_label = list(map(str, unknown_label))

            all_classes = np.unique(np.concatenate((encoder_classes, unknown_label)))
            encoder.classes_ = all_classes
        
        encoded_non_missing = encoder.transform(non_missing)
        DataFrame.loc[DataFrame[col].notna(), col] = encoded_non_missing

    joblib.dump(encoder_dict, save_path)   

    print(f'Joblib Saved\n')

    return DataFrame

def NaNColumns(DataFrame):
    """
    過濾有缺失值的欄位，後續繪製圖表觀察分佈
    """
    missing_objective_col = [col for col in DataFrame.columns if DataFrame[col].isna().sum() > 0]

    missing_col = pd.concat([DataFrame['Predicted'], DataFrame[missing_objective_col]], axis=1)

    return missing_col

def MergeAllTypeofColumns(numerical, objective):
    """
    合併Numerical、Objective欄位資料
    """
    if 'Predicted' in objective.columns:
        objective.drop(columns=['Predicted'], axis=1, inplace=True)

    all_chunk1 = pd.concat([numerical, objective], axis=1)

    return all_chunk1

def FilterColumns(DataFrame, certain_col):
    """
    過濾特定欄位或是有缺失值的欄位
    """
    filtered_col = [
        col for col in DataFrame.columns
        if col != certain_col and DataFrame[col].isna().sum() != 0
    ]

    return filtered_col

def DropSpecificRowNaN(DataFrame, col):
    """
    捨棄特定欄位的資料
    """
    print(f'Row Before Drop: {DataFrame.shape[0]}')
    drop_nan_objective = DataFrame.dropna(subset=col).reset_index(drop=True)
    print(f'Row After Drop: {drop_nan_objective.shape[0]}')

    for c in col:
        print(f'Columns {c} Missing After Drop: {drop_nan_objective[c].isna().sum()}')

    return drop_nan_objective

def ModelFillNaN(DataFrame, col):
    """
    利用決策樹學習進行缺失值填補
    """
    df_missing = DataFrame[DataFrame[col].isna()]
    df_non_missing = DataFrame[~DataFrame[col].isna()]

    x_train = df_non_missing.drop(columns=[col])
    y_train = df_non_missing[col].astype(int)
    x_missing = df_missing.drop(columns=[col])

    model = DecisionTreeClassifier(random_state=42)
    model.fit(x_train, y_train)

    predicted = model.predict(x_missing)
    df_missing[col] = predicted

    DataFrame.loc[df_missing.index, col] = df_missing[col]

    print(f'Columns {col} FillNaN is Finished\n')
    
    return DataFrame

def CheckNegativeColumns(DataFrame):
    """
    排除有負數的欄位，額外做資料調整後Skew
    """
    negative_list = []
    for col in DataFrame.columns:
        if col != 'Predicted':
            if (DataFrame[col]<0).any():
                print(f'Columns {col} including negative number')
                print(f'{DataFrame[col].min()}')
                negative_list.append(str(col))
    
    return negative_list

def SkewCountingandTransform(DataFrame, excluding):
    """
    特定欄位做Skew，拉回常態分佈，配合繪圖
    """
    filtered_df = [col for col in DataFrame.columns if col != 'Predicted']

    for col in filtered_df:
        print(f'Procseeing {col}')
            
        original_data = DataFrame[col]
        original_skew = skew(DataFrame[col])
        print(f'Original Column {col} Skew: {original_skew:.2f}')

        if col not in excluding:
            
            if original_skew > 0.5 or original_skew < -0.5:
                boxcox_transform, _ = boxcox(DataFrame[col].astype(float)+1)
                skew_boxcox = skew(boxcox_transform)

                log_transform = np.log1p(DataFrame[col])
                skew_log = skew(log_transform)

                if abs(skew_boxcox) < abs(skew_log):
                    print(f'Column {col} transform by Boxcox has lower Skew: {skew_boxcox:.2f}')
                    transformed_data = boxcox_transform
                    label = 'Box-Cox'
                    DataFrame[col] = boxcox_transform
                else:
                    print(f'Column {col} transform by Log has lower Skew: {skew_log:.2f}')
                    transformed_data = log_transform
                    label = 'Log'
                    DataFrame[col] = log_transform

                plt.figure(figsize=(5, 5))
                sns.histplot(original_data, kde=True, bins=30, color='blue', label='Original')
                plt.title(f'Original Columns {col}')
                plt.legend()
                plt.show()

                plt.figure(figsize=(5, 5))
                sns.histplot(transformed_data, kde=True, bins=30, color='orange', label=label)
                plt.title(f'After Skew Columns {col}')
                plt.legend()
                plt.show()

            else:
                print(f'Columns {col} Skew is not in range for transform')
                plt.figure(figsize=(5, 5))
                sns.histplot(original_data, kde=True, bins=30, color='blue', label='Original')
                plt.legend()
                plt.show()
        
        else:
            shifted_data = DataFrame[col] - DataFrame[col].min()
            log_transform = np.log1p(shifted_data)
            skew_log = skew(log_transform)

            print(f'Column {col} had been shifted Because it including negative: {skew_log:.2f}')
            DataFrame[col] = log_transform 

            plt.figure(figsize=(5, 5))
            sns.histplot(original_data, kde=True, bins=30, color='blue', label='Original')
            plt.title(f'Original Columns {col}')
            plt.legend()
            plt.show()

            plt.figure(figsize=(5, 5))
            sns.histplot(log_transform, kde=True, bins=30, color='orange', label=label)
            plt.title(f'After Skew Columns {col}')
            plt.legend()
            plt.show()
    
    return DataFrame

def SkewCountingandTransformWithoutHist(DataFrame, excluding):
    """
    特定欄位做Skew，拉回常態分佈，節省時間，省去繪圖
    """
    filtered_df = [col for col in DataFrame.columns if col != 'Predicted']

    for col in filtered_df:
        print(f'Procseeing {col}')
        
        original_skew = skew(DataFrame[col])
        print(f'Original Column {col} Skew: {original_skew:.2f}')

        if col not in excluding:
            
            if original_skew > 0.5 or original_skew < -0.5:
                boxcox_transform, _ = boxcox(DataFrame[col].astype(float)+1)
                skew_boxcox = skew(boxcox_transform)

                log_transform = np.log1p(DataFrame[col])
                skew_log = skew(log_transform)

                if abs(skew_boxcox) < abs(skew_log):
                    print(f'Column {col} transform by Boxcox has lower Skew: {skew_boxcox:.2f}')
                    DataFrame[col] = boxcox_transform
                else:
                    print(f'Column {col} transform by Log has lower Skew: {skew_log:.2f}')
                    DataFrame[col] = log_transform

            else:
                print(f'Columns {col} Skew is not in range for transform')
        
        else:
            shifted_data = DataFrame[col] - DataFrame[col].min()
            log_transform = np.log1p(shifted_data)
            skew_log = skew(log_transform)

            print(f'Column {col} had been shifted Because it including negative: {skew_log:.2f}')
            DataFrame[col] = log_transform 
    
    return DataFrame

def BoxPlot(DataFrame):
    """
    繪製箱型圖，觀察資料異常值
    """
    plt.figure(figsize=(30, 30))
    
    for i, col in enumerate(DataFrame.columns):
        if col != 'Predicted':
            plt.subplot(10, 4, i+1)
            sns.boxplot(x=DataFrame[col], color='blue')
            plt.title(f'Boxplot of Columns{col}')
    
    plt.tight_layout()
    plt.show()

def IQRRemoveOutlier(DataFrame, filtered_columns=None, limit_range=2.5):
    """
    利用IQR方式去除異常值
    """
    print(f'Original Shape: {DataFrame.shape}')
    for col in DataFrame.columns:
        if col != 'Predicted' and col != filtered_columns:
            Q3 = DataFrame[col].quantile(0.75)
            Q1 = DataFrame[col].quantile(0.25)
            IQR = Q3 - Q1

            lower_limit = Q1 - limit_range * IQR
            upper_limit = Q3 + limit_range * IQR

            DataFrame = DataFrame[(DataFrame[col]>=lower_limit)&(DataFrame[col]<=upper_limit)]

    print(f'After Removing Outlier Shape: {DataFrame.shape}\n')
    
    return DataFrame

def ExtremeRemoveOutlier(DataFrame, lower_quantile=0.01, upper_quantile=0.99):
    """
    刪除前後1%方式去除異常值
    """
    print(f'Original Shape: {DataFrame.shape}')
    for col in DataFrame.columns:
        if col != 'Predicted':
            lower_limit = DataFrame[col].quantile(lower_quantile)
            upper_limit = DataFrame[col].quantile(upper_quantile)

            DataFrame = DataFrame[(DataFrame[col]>=lower_limit)&(DataFrame[col]<=upper_limit)]

    print(f'After Removing Outlier Shape: {DataFrame.shape}\n')
    
    return DataFrame

def SaveCSV(DataFrame, Dtype, Number):
    files_name = f'{Dtype} Chunk {Number}'
    DataFrame.to_csv(f'{Dtype} Chunk {Number}.csv', index=False)

    print(f'Saved {files_name}')

def NumericalFunctionPackage(file_name,
                             chunk_number,
                             numerical_file_name='0.FillNan Numerical',
                             objective_file_name='1.Row Objective'):

    df_chunk = pd.read_csv(file_name, sep=',')

    df_chunk = RenameandCheckDuplicateandNaN(df_chunk)

    df_chunk = RemovedColumnOver50(df_chunk)

    numerical_chunk, objective_chunk = SeparateNumericalandObjective(df_chunk)

    numerical_chunk = ImputeNaN(numerical_chunk)

    SaveCSV(numerical_chunk, numerical_file_name, chunk_number)

    SaveCSV(objective_chunk, objective_file_name, chunk_number)

    return numerical_chunk, objective_chunk

def ObjectiveFunctionPackage(objective_chunk,
                             numerical_chunk,
                             certain_col,
                             chunk_number,
                             output_file_name='2.All FillNan'):

    objective_chunk = ObjectiveLabelEncoder(objective_chunk)

    df_chunk = MergeAllTypeofColumns(numerical_chunk, objective_chunk)

    filtered_col = FilterColumns(df_chunk, certain_col)

    df_chunk = DropSpecificRowNaN(df_chunk, filtered_col)

    df_chunk = ModelFillNaN(df_chunk, certain_col)

    SaveCSV(df_chunk, output_file_name, chunk_number)

    return df_chunk

def SkewandRemoveOutlier(df_chunk,
                         file_chunk_number,
                         numerical_col = '13',
                         objective_col = '14',
                         output_file_name='3.Final'):

    numerical_df = df_chunk.loc[:, 'Predicted':numerical_col]
    objective_df = df_chunk.loc[:, objective_col:]

    negative_col = CheckNegativeColumns(numerical_df)

    numerical_df = SkewCountingandTransformWithoutHist(numerical_df, negative_col)

    skewed_df_chunk = MergeAllTypeofColumns(numerical_df, objective_df)

    SaveCSV(skewed_df_chunk, output_file_name, file_chunk_number)

    return skewed_df_chunk

def ReadandMergeAllChunk(last_chunk_number):

    chunks = []

    for i in range(1, last_chunk_number+1):
        file_name = f'3.Final Chunk {i}.csv'
        if os.path.exists(file_name):
            chunk = pd.read_csv(file_name, sep=',')
            chunks.append(chunk)
            print(f'{file_name} is Loaded')
        else:
            print(f'{file_name} is not Found')

    merge_df = pd.concat(chunks, ignore_index=True)
    print('All Chunk is Read and Merged\n')

    return merge_df

def DataFrameStandardScaler(DataFrame, exclude_col='Predicted'):

    filtered_df = [col for col in DataFrame.columns if col != exclude_col]

    scaler = StandardScaler()

    scaler_df = scaler.fit_transform(DataFrame[filtered_df])

    standardized_df = pd.DataFrame(scaler_df, columns=filtered_df)

    final_df = pd.concat([DataFrame[exclude_col], standardized_df], axis=1)

    return final_df

def ScatterPlot(DataFrame):

    plt.figure(figsize=(20, 20))

    for i, col in enumerate(DataFrame.columns):
        if col != 'Predicted':
            plt.subplot(10, 4, i+1)
            sns.scatterplot(data=DataFrame, x='Predicted', y=col)
            plt.title(f'Predicetd vs {col}')
    
    plt.tight_layout()
    plt.show()

def TestTrainSplit(DataFrame, test_size=0.2):

    x = DataFrame.drop(columns=['Predicted'])
    y = DataFrame['Predicted']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test

def RandomForestFeature(withoutlabel, label, estimator_list, x_train, y_train):

    plt.figure(figsize=(20, 20))
    for i, estimator in enumerate(estimator_list):
        rf_model = RandomForestRegressor(n_estimators=estimator, random_state=42)
        rf_model.fit(x_train, y_train)
        rf_importances = rf_model.feature_importances_
        rf_feature = withoutlabel.columns

        feature_importances = pd.Series(rf_importances, index=rf_feature)
        top_features = feature_importances.head().index

        print(top_features)

        plt.subplot(10, 4, i+1)
        plt.barh(rf_feature, rf_importances)
        plt.title(f'Random Forest {estimator}')

    plt.tight_layout()
    plt.show()

def PCAFeature(DataFrame, n_components=2):

    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(DataFrame)

    plt.scatter(pca_result[:, 0], pca_result[:, 1], c='blue')
    plt.title('PCA Visualization')
    plt.show()

def SMOTEReSample(x_train, y_train, random_state=42):

    resample_x_train, resample_y_train = SMOTE(random_state=random_state).fit_resample(x_train, y_train)

    return resample_x_train, resample_y_train

if __name__ == '__main__':
    print(f'This is Def Function Script')