import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, roc_auc_score, precision_score, recall_score, f1_score, roc_curve, accuracy_score
from sklearn.model_selection import KFold
import optuna
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Dropout, concatenate, Lambda, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def LoadandProcessData(train_file, test_file, numeric_cols):
    df_train = pd.read_csv(train_file, sep=',')
    df_test = pd.read_csv(test_file, sep=',')
    df = pd.concat([df_train, df_test], ignore_index=True)

    feature_dict = {}
    total_feature = 0
    for col in df.columns:
        if col == 'Predicted':
            continue
        elif col in numeric_cols:
            feature_dict[col] = total_feature
            total_feature += 1
        else:
            unique_val = df[col].unique()
            feature_dict[col] = dict(zip(unique_val, range(total_feature, len(unique_val) + total_feature)))
            total_feature += len(unique_val)

    return df_train, df_test, feature_dict, total_feature

def PreprocessingData(DataFrame, feature_dict, numric_cols):
    feature_index = DataFrame.copy()
    feature_value = DataFrame.copy()

    for col in DataFrame.columns:
        if col in numric_cols:
            feature_index[col] = feature_dict[col]
        else:
            feature_index[col] = feature_index[col].map(feature_dict[col])
            feature_value[col] = 1
    
    return feature_index, feature_value

def build_deepfm(feature_size, field_size, params):
    feat_index = Input(shape=(field_size,), dtype=tf.int32, name='feat_index')
    feat_value = Input(shape=(field_size,), dtype=tf.float32, name='feat_value')

    embeddings = Embedding(input_dim=feature_size, output_dim=params['embedding_size'])(feat_index)
    feat_value_reshaped = Lambda(lambda x: tf.expand_dims(x, axis=-1))(feat_value)
    embeddings = Multiply()([embeddings, feat_value_reshaped])

    summed_features_emb = Lambda(lambda x: tf.reduce_sum(x, axis=1))(embeddings)
    summed_features_emb_square = Lambda(lambda x: tf.square(x))(summed_features_emb)

    squared_features_emb = Lambda(lambda x: tf.square(x))(embeddings)
    squared_sum_features_emb = Lambda(lambda x: tf.reduce_sum(x, axis=1))(squared_features_emb)

    fm_second_order = Lambda(lambda x: 0.5 * tf.subtract(x[0], x[1]))(
    [summed_features_emb_square, squared_sum_features_emb]
    )

    y_deep = Flatten()(embeddings)
    for i, units in enumerate(params['deep_layers']):
        y_deep = Dense(units=units, activation='relu')(y_deep)
        y_deep = Dropout(params['dropout_deep'][i])(y_deep)

    concat_input = concatenate([summed_features_emb, fm_second_order, y_deep])
    output = Dense(1, activation='sigmoid')(concat_input)

    model = Model(inputs=[feat_index, feat_value], outputs=output)

    return model

def objective_data(trial, feature_index, feature_value, train_y, total_feature):
    embedding_size = trial.suggest_int("embedding_size", 4, 32)
    deep_layers = trial.suggest_categorical("deep_layers", [[32, 32], [64, 32], [64, 64]])
    dropout_deep = trial.suggest_float("dropout_deep", 0.1, 0.7, step=0.2)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
    epochs = trial.suggest_int("epochs", 5, 15)

    dfm_params = {
        "embedding_size":embedding_size,
        "deep_layers":deep_layers,
        "dropout_deep":[dropout_deep] * len(deep_layers),
        "batch_size":batch_size,
        "learning_rate":learning_rate,
        "optimizer":"adam",
        "loss":'binary_crossentropy',
        "metrics":["AUC"],
        "epochs":epochs
    }

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    log_loss_scores = []

    for train_idx, val_idx in kfold.split(feature_index):
        train_feature_index, val_feature_index = feature_index.iloc[train_idx], feature_index.iloc[val_idx]
        train_feature_value, val_feature_value = feature_value.iloc[train_idx], feature_value.iloc[val_idx]
        train_y_cv, val_y_cv = train_y[train_idx], train_y[val_idx]

        deepfm_model = build_deepfm(
            feature_size=total_feature,
            field_size=feature_index.shape[1],
            params=dfm_params
        )

        deepfm_model.compile(
            optimizer=tf.keras.optimizers.get(
                {"class_name": dfm_params['optimizer'], "config": {"learning_rate": dfm_params["learning_rate"]}}
            ),
            loss=dfm_params["loss"],
            metrics=dfm_params["metrics"]
        )

        deepfm_model.fit(
            x=[train_feature_index, train_feature_value],
            y=train_y_cv,
            validation_data=([val_feature_index, val_feature_value], val_y_cv),
            batch_size=dfm_params["batch_size"],
            epochs=dfm_params["epochs"],
            verbose=1
        )

        y_pred = deepfm_model.predict([val_feature_index, val_feature_value])
        logloss = log_loss(val_y_cv, y_pred)
        log_loss_scores.append(logloss)

    return np.mean(log_loss_scores)

def kfold_train_and_evaluate(feature_index, feature_value, train_y, test_feature_index, test_feature_value, test_y, params, total_feature, field_size, k=5):
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    log_loss_scores = []
    auc_scores = []
    model_weights = []

    for train_idx, val_idx in kfold.split(feature_index):
        train_feature_index, val_feature_index = feature_index.iloc[train_idx], feature_index.iloc[val_idx]
        train_feature_value, val_feature_value = feature_value.iloc[train_idx], feature_value.iloc[val_idx]
        train_y_fold, val_y_fold = train_y[train_idx], train_y[val_idx]

        model = build_deepfm(
            feature_size=total_feature,
            field_size=field_size,
            params=params
        )

        model.compile(
            optimizer=tf.keras.optimizers.get(
                {"class_name": params["optimizer"], "config": {"learning_rate": params["learning_rate"]}}
            ),
            loss=params["loss"],
            metrics=params["metrics"]
        )

        model.fit(
            x=[train_feature_index, train_feature_value],
            y=train_y_fold,
            validation_data=([val_feature_index, val_feature_value], val_y_fold),
            batch_size=params["batch_size"],
            epochs=params["epochs"],
            verbose=1
        )

        y_val_pred = model.predict([val_feature_index, val_feature_value])
        logloss = log_loss(val_y_fold, y_val_pred)
        auc = roc_auc_score(val_y_fold, y_val_pred)
        log_loss_scores.append(logloss)
        auc_scores.append(auc)

        model_weights.append(model.get_weights())

    averaged_model = build_deepfm(
        feature_size=total_feature,
        field_size=field_size,
        params=params
    )
    averaged_weights = [np.mean([weights[i] for weights in model_weights], axis=0) for i in range(len(model_weights[0]))]
    averaged_model.set_weights(averaged_weights)

    y_test_pred = averaged_model.predict([test_feature_index, test_feature_value])
    y_test_pred_binary = (y_test_pred > 0.5).astype(int)

    test_logloss = log_loss(test_y, y_test_pred)
    test_auc = roc_auc_score(test_y, y_test_pred)
    test_accuracy = accuracy_score(test_y, y_test_pred_binary)
    test_precision = precision_score(test_y, y_test_pred_binary)
    test_recall = recall_score(test_y, y_test_pred_binary)
    test_f1 = f1_score(test_y, y_test_pred_binary)

    print(f"Average Validation Log Loss: {np.mean(log_loss_scores):.3f}")
    print(f"Average Validation AUC: {np.mean(auc_scores):.3f}")
    print(f"Test Log Loss: {test_logloss:.3f}")
    print(f"Test AUC: {test_auc:.3f}")
    print(f"Test Accuracy: {test_accuracy:.3f}")
    print(f"Test Precision: {test_precision:.3f}")
    print(f"Test Recall: {test_recall:.3f}")
    print(f"Test F1 Score: {test_f1:.3f}")

    return averaged_model

if __name__ == "__main__":

    x_train_file = '/Users/porter/Desktop/01.Data Science/Porter - Project/05 - Advertising Challenge - Data Analysis/Data/X_train.csv'
    y_train_file = '/Users/porter/Desktop/01.Data Science/Porter - Project/05 - Advertising Challenge - Data Analysis/Data/Y_train.csv'
    x_test_file = '/Users/porter/Desktop/01.Data Science/Porter - Project/05 - Advertising Challenge - Data Analysis/Data/X_test.csv'
    y_test_file = '/Users/porter/Desktop/01.Data Science/Porter - Project/05 - Advertising Challenge - Data Analysis/Data/Y_test.csv'

    Numeric_cols = ["1", "3", "4", "5", "6", "7", "10", "11", "13"]

    df_train, df_test, feature_dict, total_feature = LoadandProcessData(x_train_file, y_train_file, Numeric_cols)
    feature_index, feature_value = PreprocessingData(df_train, feature_dict, Numeric_cols)
    train_y = df_test['Predicted'].values

    evaluate_train, evaluate_test, evaluate_feature_dict, evaluate_total_feature = LoadandProcessData(x_test_file, y_test_file, Numeric_cols)
    evaluate_feature_index, evaluate_feature_value = PreprocessingData(evaluate_train, evaluate_feature_dict, Numeric_cols)
    test_y = evaluate_test["Predicted"].values

    def objective(trial):
        return objective_data(trial, feature_index, feature_value, train_y, total_feature)
    
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    print("Best Parameters:", study.best_params)
    print("Best Log Loss:", study.best_value)

    best_params = {
        "embedding_size": 10,
        "deep_layers": [64, 32],
        "dropout_deep": [0.1, 0.1],
        "batch_size": 1024,
        "learning_rate": 0.00034021406978301225,
        "optimizer": "adam",
        "loss": "binary_crossentropy",
        "metrics": ["AUC"],
        "epochs": 15
    }

    dfm_params = {
        "embedding_size": best_params["embedding_size"],
        "deep_layers": best_params["deep_layers"],
        "dropout_deep": [best_params["dropout_deep"]] * len(best_params["deep_layers"]),
        "batch_size": best_params["batch_size"],
        "learning_rate": best_params["learning_rate"],
        "optimizer": "adam",
        "loss": "binary_crossentropy",
        "metrics": ["AUC"],
        "epochs": best_params["epochs"],
    }

    final_model = kfold_train_and_evaluate(
        feature_index=feature_index,
        feature_value=feature_value,
        train_y=train_y,
        test_feature_index=evaluate_feature_index,
        test_feature_value=evaluate_feature_value,
        test_y=test_y,
        params=best_params,
        total_feature=total_feature,
        field_size=feature_index.shape[1]
    )

    final_model.save("best_deepfm_model.h5")
    print("Final model saved as 'best_deepfm_model.h5'")