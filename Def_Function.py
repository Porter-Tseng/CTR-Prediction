import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from scipy.stats import skew, boxcox, pearsonr, pointbiserialr, zscore
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import log_loss

class EDAFunction:
    def __init__(self):
        pass

    def ImportandSeparate(self, file_name, chunksize, sep="\t"):
        read_chunk = pd.read_csv(file_name, sep=sep, chunksize=chunksize, iterator=True)

        total_row = 0
        total_missing_row = 0

        for i, chunk in enumerate(read_chunk, start=1):
            print(f"Processing Chunk{i}")
            print(f"Chunk{i} Shape: {chunk.shape}")
            total_row += chunk.shape[0]

            missing_rows = chunk.isnull().any(axis=1).sum()
            total_missing_row += missing_rows

            chunk.to_csv(f"Base Chunk{i}.csv", index=False)

            print(f"Chunk{i} Finished")
        
        print("---Summary----")
        print(f"Total Rows in Data: {total_row}")
        print(f"Total Missing Row: {total_missing_row}")

    def LabelValueCount(self, data, target="0"):
        value_count = data[target].value_counts(normalize=True)*100
        formatted_count = value_count.round(2).astype(str)+"%"

        print(formatted_count)

    def ColumnsNaNSorted(self, data):
        nan_percentage = (data.isna().sum() / data.shape[0]*100).sort_values(ascending=False)
        filtered_nan = nan_percentage[nan_percentage != 0]
        print(f"{filtered_nan}\n")

    def ColumnsHist(self, data, filter_include="number"):
        filter_col = data.select_dtypes(include=filter_include).columns
        plt.figure(figsize=(25, 40))

        for i, col in enumerate(filter_col):

            plt.subplot(10, 4, i+1)

            if col == "Predicted":
                col_value_count = data[col].value_counts()
                plt.pie(col_value_count, labels=col_value_count.index, autopct='%1.1f%%')

                plt.title(f"Proporiton of {col}")
            else:
                cleaned_data = data[col].dropna()

                sns.histplot(cleaned_data, bins=30, kde=True, color="blue")

                plt.title(f"Distribution of Columns{col}")
                plt.xlabel("Values")
                plt.ylabel("Amount")

        plt.tight_layout()
        plt.show()

    def CorrHeatMap(self, data, method="spearman", figsize=(10, 10)):
        corr_col = data.corr(method=method)

        plt.figure(figsize=figsize)
        sns.heatmap(
            corr_col,
            annot=True,
            annot_kws={"size":8},
            cmap="coolwarm",
            fmt=".2f"
        )
        plt.title(f"Correlation Heatmap ({method.capitalize()})")
        plt.show()
    
    def Process_Chunk(self, input_file_name, output_name, import_path, output_path, first_range=1, last_range=13, sep=","):

        os.chdir(import_path)

        if not os.path.exists(input_file_name):
            print(f"Chunk File Not Found: {input_file_name}")
            return
        
        print(f"Processing {input_file_name} ...\n")

        data = pd.read_csv(input_file_name, sep=",")

        duplicated_row = 0
        missing_row = 0
        missing_rows = data.isnull().any(axis=1).sum()
        missing_row += missing_rows

        data = self.RenameColumns(data)
        data, duplicates_number = self.RemoveDuplicates(data)
        duplicated_row += duplicates_number
        data = self.RemoveHighPortionColumns(data, threshold=50)
        data = self.ValueLabelEncoder(data)

        numeric_col = data.select_dtypes(include="number").columns
        data = self.ImputeNan(data, numeric_col, strategy="median")
        data = self.DropRowNan(data)
        print(f"Check is NaN Number: {data.isna().sum().all()}\n")

        data = self.SkewColumns(data, first_range, last_range)
        
        os.chdir(output_path)
        self.SaveCSV(data, output_name)

        return duplicated_row, missing_row

    def RenameColumns(self, data):
        data.columns = ['Predicted']+[str(i) for i in range(1, len(data.columns))]
        print(f'Rename is Finished.\n')
        return data
    
    def RemoveDuplicates(self, data):
        initial_shape = data.shape
        duplicates_number = data.duplicated().sum()
        data.drop_duplicates(inplace=True)
        print(f"Original Shape: {initial_shape} -> New Shape: {data.shape}")
        print(f"Duplicates Number: {duplicates_number}\n")
        return data, duplicates_number
    
    def CountStatistic(self, data, include="number"):
        filter_col = data.select_dtypes(include=include).columns

        for col in filter_col:
            cleaned_col = data[col].dropna()

            col_mode = cleaned_col.mode().iloc[0]
            col_median = cleaned_col.median()    
            col_avg = round(cleaned_col.mean(), 3)

            print(f'Statistic of Columns{col}')
            print(f'Mode: {col_mode}')
            print(f'Median: {col_median}')
            print(f'AVG: {col_avg}\n')

    def RemoveHighPortionColumns(self, data, threshold=50):
        nan_percentage = data.isna().sum()/data.shape[0]*100
        columns_drop = nan_percentage[nan_percentage>threshold].index
        data.drop(columns=columns_drop, inplace=True)
        print(f"Removed Columns Which with {threshold}% of Nan: {list(columns_drop)}\n")
        return data
    
    def ImputeNan(self, data, filter_columns, strategy="median"):
        imputer = SimpleImputer(strategy=strategy)
        for col in filter_columns:
            if data[col].isna().sum() > 0:
                data[col] = pd.Series(
                    imputer.fit_transform(data[[col]]).ravel(),
                    index=data.index
                )
        print(f"Missing Value Had Imputed by {strategy}\n")

        return data
    
    def DropRowNan(self, data):
        nan_columns = [col for col in data.columns if data[col].isna().sum()>0]
        print(f"Row Before Drop: {data.shape[0]}")
        drop_nan_data = data.dropna(subset=nan_columns).reset_index(drop=True)
        print(f"Row After Drop: {drop_nan_data.shape[0]}\n")

        return drop_nan_data
    
    def ValueLabelEncoder(self, data, save_path="encoder_dict.joblib"):
        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            encoder_dict = joblib.load(save_path)
            print("Success Load encoder_dict")
        else:
            print("encoder_dict is None")
            encoder_dict = {}

        objective_col = data.select_dtypes(include="object").columns

        for col in objective_col:
            if col in encoder_dict:
                encoder = encoder_dict[col]
            else:
                encoder = LabelEncoder()
                encoder.fit(data[col].dropna().astype(str))
                encoder_dict[col] = encoder
            
            unknown_label = set(data[col].dropna().astype(str)) - set(encoder.classes_)
            if unknown_label:
                all_classes = list(encoder.classes_) + list(unknown_label)
                encoder.classes_ = np.array(sorted(all_classes))

            non_missing_mask = data[col].notna()
            data.loc[non_missing_mask, col] = encoder.transform(data.loc[non_missing_mask, col].astype(str))

        joblib.dump(encoder_dict, save_path)   
        print(f"Encoder Dict Joblib Saved\n")

        return data

    def NormalizedScaler(self, data, first_range=1, last_range=13, label=None):
        filtered_df = data.iloc[:, first_range:last_range]

        scaler = MinMaxScaler()

        scaler_df = scaler.fit_transform(filtered_df)

        standardized_df = pd.DataFrame(scaler_df, columns=filtered_df.columns)

        objective_col = data.select_dtypes(include="object")

        if label is not None:
            final_df = pd.concat([data[label], standardized_df, objective_col], axis=1)
        else:
            final_df = pd.concat([standardized_df, objective_col], axis=1)

        return final_df
    
    def SkewColumns(self, data, first_range=1, last_range=14):
        
        data = data.copy()

        numerical_columns = data.iloc[:, first_range:last_range]

        for col in numerical_columns.columns:

            data[col] = data[col] - data[col].min() + 1

            transformed_data, _ = boxcox(data[col])

            data[col] = transformed_data

        print(f"Skew Finished\n")
        
        return data
    
    def SaveCSV(self, data, file_name):
        data.to_csv(f"{file_name}.csv", index=False)
        print(f"Saved {file_name}")

    def ReadandMergeAllChunk(self, last_chunk_number, file_name="Filled Chunk{}.csv"):
        chunks = []

        for i in range(1, last_chunk_number+1):
            formatted_file_name = file_name.format(i)
            if os.path.exists(formatted_file_name):
                chunk = pd.read_csv(formatted_file_name, sep=',')
                chunks.append(chunk)
                print(f'{formatted_file_name} is Loaded')
            else:
                print(f'{formatted_file_name} is not Found')

        data = pd.concat(chunks, ignore_index=True)
        print('All Chunk is Read and Merged\n')

        return data
    
    def IQRRemoveOutlier(self, data, filtered_columns=None, limit_range=1.5):
        print(f"Original Shape: {data.shape}")
        for col in data.columns:
            if col != "Predicted" and col != filtered_columns:
                Q3 = data[col].quantile(0.75)
                Q1 = data[col].quantile(0.25)
                IQR = Q3 - Q1

                lower_limit = Q1 - limit_range * IQR
                upper_limit = Q3 + limit_range * IQR

                data = data[(data[col]>=lower_limit)&(data[col]<=upper_limit)]
        print(f"After Removing Outlier Shape: {data.shape}\n")
        
        return data
    
    def ExtremeRemoveOutlier(self, data, lower_quantile=0.01, upper_quantile=0.99, exclude=["Predicted", "22", "30"]): 
        print(f"Original Shape: {data.shape}")

        for col in data.columns:
            if col not in exclude:
                lower_limit = data[col].quantile(lower_quantile)
                upper_limit = data[col].quantile(upper_quantile)

                data = data[(data[col]>=lower_limit)&(data[col]<=upper_limit)]

        print(f"After Removing Outlier Shape: {data.shape}\n")
        
        return data
    
    def RandomForestFeature(self, data, estimator_list, targeted_col="Predicted", max_depth=10):
        x = data.drop(columns=[targeted_col])
        y = data[targeted_col]
        rf_feature = x.columns
        feature_dict = {}

        for estimator in estimator_list:
            rf_model = RandomForestRegressor(
                n_estimators=estimator,
                max_depth=max_depth,
                max_features=10,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(x, y)

            feature_importances = pd.Series(rf_model.feature_importances_, index=rf_feature)
            top_features = feature_importances.sort_values(ascending=False)
            print(f"Top Feature for {estimator} Estimators:")
            print(f"{top_features.to_string(index=True)}\n")

            feature_dict[estimator] = top_features

        return feature_dict

    def SelectTopCorrFeature(self, data, target_col, method="spearman", top_percent=0.5):
        corr = data.corr(method=method)
        corr_target = corr[target_col].drop(target_col)
        corr_abs = corr_target.abs()

        result = {}
        sorted_dfs = {}

        for percent in top_percent:
            feature_number = int(len(corr_abs)*percent)
            top_feature = corr_abs.sort_values(ascending=False).head(feature_number).index.tolist()
            result[percent] = top_feature

            print(f"Top {percent*100:.0f}% Correlated Feature: {top_feature}\n")

            sorted_columns = [target_col] + sorted(top_feature, key=int)
            sorted_dfs[percent] = data[sorted_columns]
        
        return result, sorted_dfs
    
    def PCAdecomposition(self, data, n_components=2):
        pca = PCA(n_components=n_components)

        pca_data = pca.fit_transform(data)

        return pca_data

if __name__ == '__main__':
    print(f'This is Def Function Script')