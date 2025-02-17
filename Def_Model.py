import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, roc_auc_score, precision_score, accuracy_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, Flatten, Dropout, concatenate, Lambda, Multiply
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam, AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import optuna
from optuna.integration import TFKerasPruningCallback
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

def TestTrainSplit(data_file, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, sample_ratio=1.0, random_state=42):
    assert train_ratio + val_ratio + test_ratio == 1

    data = pd.read_csv(data_file, sep=",")

    if sample_ratio < 1.0:
        data = data.sample(frac=sample_ratio, random_state=random_state)
        print(f"Sample Data Shape: {data.shape}")
    
    x = data.drop(columns=["Predicted"])
    y = data[["Predicted"]]

    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=(1-train_ratio), random_state=random_state)

    if test_ratio == 0:
        x_test, y_test = np.array([]), np.array([])
        x_val, y_val = x_temp, y_temp
    else:
        x_val, x_test, y_val, y_test = train_test_split(
            x_temp, y_temp,
            test_size=(val_ratio / (val_ratio + test_ratio)),
            random_state=random_state,
            stratify=y_temp
        )

    return x_train, x_val, x_test, y_train, y_val, y_test

def GenerateFeatureAndDict(train, validation, test, numeric_cols):
    data = pd.concat([train, validation, test])

    feature_dict = {}
    total_feature = 0
    label_encoders = {}
    scalers = {}

    for col in data.columns:    
        if col == 'Predicted':
            continue
        elif col in numeric_cols:
            scaler = MinMaxScaler()
            data[col] = scaler.fit_transform(data[[col]])
            scalers[col] = scaler

            feature_dict[col] = total_feature
            total_feature += 1
        else:
            label_encoder = LabelEncoder()
            encoded_col = label_encoder.fit_transform(data[col].astype(str))
            label_encoders[col] = label_encoder
            
            unique_val = np.unique(encoded_col)
            feature_dict[col] = {v: i for i, v in enumerate(unique_val, start=total_feature)}
            total_feature += len(unique_val)

            data[col] = encoded_col

    return feature_dict, total_feature, label_encoders, scalers

def DataTransfer(data, feature_dict, numeric_cols, scalers):

    feature_index = data.copy()
    feature_value = data.copy()

    for col in data.columns:
        if col in numeric_cols:
            feature_index[col] = feature_dict[col]
            feature_value[col] = scalers[col].transform(data[[col]])
        else:
            feature_index[col] = feature_index[col].map(feature_dict[col])
            feature_value[col] = 1

    feature_index_np = feature_index.to_numpy(dtype=np.int32)
    feature_value_np = feature_value.to_numpy(dtype=np.float32)
    
    return feature_index_np, feature_value_np

def BuildDeepFMModel(feature_size, field_size, params):
    feat_index = Input(shape=(field_size,), dtype=tf.int32, name="feat_index")
    feat_value = Input(shape=(field_size,), dtype=tf.float32, name="feat_value")

    embeddings = Embedding(
        input_dim=feature_size,
        output_dim=params["embedding_size"],
        embeddings_regularizer=l2(params["l2_reg"])
    )(feat_index)
    feat_value_reshaped = Lambda(lambda x: tf.expand_dims(x, axis=-1))(feat_value)
    embeddings = Multiply()([embeddings, feat_value_reshaped])

    summed_features_emb = Lambda(lambda x: tf.reduce_sum(x, axis=1))(embeddings)
    summed_features_emb_square = Lambda(lambda x: tf.square(x))(summed_features_emb)

    squared_feature_emb = Lambda(lambda x: tf.square(x))(embeddings)
    squared_feature_emb_summed = Lambda(lambda x: tf.reduce_sum(x, axis=1))(squared_feature_emb)
    
    fm_second_order = Lambda(lambda x: 0.5 * tf.subtract(x[1], x[0]))(
        [summed_features_emb_square, squared_feature_emb_summed]
    )

    y_deep = Flatten()(embeddings)
    for i, units in enumerate(params["deep_layers"]):
        y_deep = Dense(
            units=units,
            kernel_regularizer=l2(params["l2_reg"]),
            use_bias=not params.get("batch_norm", 0)
        )(y_deep)

        if params.get("batch_norm", 0) == 1:
            y_deep = BatchNormalization(momentum=params.get("batch_norm_decay", 0))(y_deep)
            
        y_deep = Activation("relu")(y_deep)
        y_deep = Dropout(params["dropout_deep"][i])(y_deep)

    concat_input = concatenate([summed_features_emb, fm_second_order, y_deep])
    output = Dense(
        1,
        activation="sigmoid",
        kernel_regularizer=l2(params["l2_reg"])
    )(concat_input)

    model = Model(inputs=[feat_index, feat_value], outputs=output)

    return model

def OptunaTrainModel(
        trial, feature_index, feature_value, train_y,
        val_feature_index, val_feature_value, val_y,
        total_feature, earlystopping_patience, epochs,
        n_split, predict_threshold
    ):
    embedding_size = trial.suggest_int("embedding_size", 8, 16)
    dropout_fm = trial.suggest_categorical("dropout_fm", [[0.4, 0.4], [0.3, 0.3], [0.2, 0.2], [0.1, 0.1]])
    deep_layers = trial.suggest_categorical("deep_layers", [(64, 32), (128, 64), (128, 128), (256, 128)])
    dropout_deep = trial.suggest_float("dropout_deep", 0.2, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [4096, 8192, 16384])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["sgd", "adam", "adamw"])
    l2_reg = trial.suggest_float("l2_reg", 1e-5, 5e-3, log=True)
    batch_norm_decay = trial.suggest_float("batch_norm_decay", 0.9, 0.98)

    epochs = epochs
    batch_norm = 1

    params = {
        "embedding_size":embedding_size,
        "dropout_fm":dropout_fm,
        "deep_layers":deep_layers,
        "dropout_deep":[dropout_deep] * len(deep_layers),
        "batch_size":batch_size,
        "learning_rate":learning_rate,
        "optimizer":optimizer_name,
        "loss":'binary_crossentropy',
        "metrics":[tf.keras.metrics.AUC(name="auc")],
        "epochs":epochs,
        "l2_reg":l2_reg,
        "batch_norm": batch_norm,
        "batch_norm_decay": batch_norm_decay
    }

    earlystopping = EarlyStopping(monitor="val_loss", patience=earlystopping_patience, restore_best_weights=True)
    pruning_callback = TFKerasPruningCallback(trial, monitor="val_loss")

    if params["optimizer"] == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=params["learning_rate"], momentum=0.9, nesterov=True)
    elif params["optimizer"] == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"])
    else:
        optimizer = tf.keras.optimizers.AdamW(learning_rate=params["learning_rate"], weight_decay=params.get("l2_reg", 0.0))
    
    if n_split == 0:
        dpfm = BuildDeepFMModel(
        feature_size=total_feature,
        field_size=feature_index.shape[1],
        params=params
        )

        dpfm.compile(
        optimizer=optimizer,
        loss=params["loss"],
        metrics=params["metrics"]
        )

        dpfm.fit(
            x=[feature_index, feature_value],
            y=train_y,
            batch_size=params["batch_size"],
            epochs=params["epochs"],
            validation_data=([val_feature_index, val_feature_value], val_y),
            callbacks=[earlystopping, pruning_callback],
            verbose=1
        )

        y_pred = dpfm.predict([val_feature_index, val_feature_value])
        y_pred_binary = (y_pred >= predict_threshold).astype(int)

        fpr, tpr, _ = roc_curve(val_y, y_pred)

        plt.figure(figsize=(10, 10))
        plt.plot(fpr, tpr, color="red", label=f"ROC (AUC = {roc_auc_score(val_y, y_pred):.2f})")
        plt.title("ROC Curve with AUC")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.grid()
        plt.show()

        return {
            "log loss": round(log_loss(val_y, y_pred), 3),
            "roc auc": round(roc_auc_score(val_y, y_pred), 3),
            "precision": round(precision_score(val_y, y_pred_binary), 3),
            "accuracy": round(accuracy_score(val_y, y_pred_binary), 3),
            "recall": round(recall_score(val_y, y_pred_binary), 3),
            "f1 score": round(f1_score(val_y, y_pred_binary), 3)
        }

    else:
        kfold = KFold(n_splits=n_split, shuffle=True, random_state=42)
        log_loss_score_ls, auc_score_ls, precsion_score_ls = [], [], []
        accuracy_score_ls, recall_score_ls, f1_score_ls = [], [], []
        fpr_ls, tpr_ls = [], []

        for train_idx, val_idx in kfold.split(train_y):
            train_feature_index, val_feature_index = feature_index[train_idx], feature_index[val_idx]
            train_feature_value, val_feature_value = feature_value[train_idx], feature_value[val_idx]
            train_y_fold, val_y_fold = train_y[train_idx], train_y[val_idx]

            dpfm = BuildDeepFMModel(
                feature_size=total_feature,
                field_size=feature_index.shape[1],
                params=params
            )

            dpfm.compile(
                optimizer=optimizer,
                loss=params["loss"],
                metrics=params["metrics"]
            )

            dpfm.fit(
                x=[train_feature_index, train_feature_value],
                y=train_y_fold,
                batch_size=params["batch_size"],
                epochs=params["epochs"],
                validation_data=([val_feature_index, val_feature_value], val_y_fold),
                callbacks=[earlystopping, pruning_callback],
                verbose=1
            )
            
            y_pred = dpfm.predict([val_feature_index, val_feature_value])
            y_pred_binary = (y_pred >= predict_threshold).astype(int)

            log_loss_score_ls.append(log_loss(val_y_fold, y_pred))
            auc_score_ls.append(roc_auc_score(val_y_fold, y_pred))
            precsion_score_ls.append(precision_score(val_y_fold, y_pred_binary))
            accuracy_score_ls.append(accuracy_score(val_y_fold, y_pred_binary))
            recall_score_ls.append(recall_score(val_y_fold, y_pred_binary))
            f1_score_ls.append(f1_score(val_y_fold, y_pred_binary))

            fpr, tpr, _ = roc_curve(val_y_fold, y_pred)
            fpr_ls.append(fpr)
            tpr_ls.append(tpr)

        all_fpr = np.unique(np.concatenate([fpr for fpr in fpr_ls]))
        mean_tpr = np.zeros_like(all_fpr)

        for i in range(len(fpr_ls)):
            mean_tpr += np.interp(all_fpr, fpr_ls[i], tpr_ls[i])
        mean_tpr /= len(fpr_ls)

        plt.figure(figsize=(10, 10))
        plt.plot(all_fpr, mean_tpr, color="red", label=f"Mean ROC (AUC = {np.mean(auc_score_ls):.2f})")
        plt.title("ROC Curve with AUC")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.grid()
        plt.show()

def CompileAndFitModel(
        feature_index, feature_value, train_y,
        val_feature_index, val_feature_value, val_y,
        params, total_feature, earlystopping_patience,
        reduce_factor=0.7, reduce_patience=3, min_lr=1e-7,
        decay_steps=2000, decay_rate=0.96, n_split=0
    ):
    earlystopping = EarlyStopping(monitor="val_loss", patience=earlystopping_patience, restore_best_weights=True)

    if params.get("lr_strategy") == "reduce_lr":
        lr_callback = ReduceLROnPlateau(
            monitor="val_loss",
            factor=reduce_factor,
            patience=reduce_patience,
            min_lr=min_lr,
            verbose=1
        )
        learning_rate = params["learning_rate"]
    elif params.get("lr_strategy") == "exp_decay":
        lr_callback = None
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=params["learning_rate"],
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=True
        )
    else:
        lr_callback = None
        learning_rate = params["learning_rate"] 

    if params["optimizer"] == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
    elif params["optimizer"] == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=params.get("l2_reg", 0.0))
    
    if n_split == 0:
        dpfm_model = BuildDeepFMModel(
            feature_size=total_feature,
            field_size=feature_index.shape[1],
            params=params
        )

        dpfm_model.compile(
                optimizer=optimizer,
                loss=params["loss"],
                metrics=params["metrics"]
            )
        
        callbacks = [earlystopping]
        if lr_callback is not None:
            callbacks.append(lr_callback)

        history = dpfm_model.fit(
            x=[feature_index, feature_value],
            y=train_y,
            batch_size=params["batch_size"],
            epochs=params["epochs"],
            validation_data=([val_feature_index, val_feature_value], val_y),
            callbacks=callbacks,
            verbose=1
        )

        return dpfm_model, history
    
    else:
        kfold = KFold(n_splits=n_split, shuffle=True, random_state=42)

        histories = []
        best_model = None
        best_auc = 0
        best_loss = float("inf")

        for fold, (train_idx, val_idx) in enumerate(kfold.split(train_y)):

            print(f"Training Fold{fold+1} ...\n")

            tf.keras.backend.clear_session()

            train_feature_index, val_feature_index = feature_index[train_idx], feature_index[val_idx]
            train_feature_value, val_feature_value = feature_value[train_idx], feature_value[val_idx]
            train_y_fold, val_y_fold = train_y[train_idx], train_y[val_idx]

            if params["optimizer"] == "sgd":
                optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
            elif params["optimizer"] == "adam":
                optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            else:
                optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=params.get("l2_reg", 0.0))

            dpfm_model = BuildDeepFMModel(
                feature_size=total_feature,
                field_size=feature_index.shape[1],
                params=params
            )

            dpfm_model.compile(
                optimizer=optimizer,
                loss=params["loss"],
                metrics=params["metrics"]
            )
            
            callbacks = [earlystopping]
            if lr_callback is not None:
                callbacks.append(lr_callback)

            history = dpfm_model.fit(
                x=[train_feature_index, train_feature_value],
                y=train_y_fold,
                batch_size=params["batch_size"],
                epochs=params["epochs"],
                validation_data=([val_feature_index, val_feature_value], val_y_fold),
                callbacks=callbacks,
                verbose=1
            )

            histories.append(history)

            val_loss, val_auc = dpfm_model.evaluate([val_feature_index, val_feature_value], val_y_fold, verbose=1)
            print(f"\nFold {fold + 1} - Validation AUC: {val_auc:.3f}, Validation Log Loss: {val_loss:.3f}\n")

            if val_auc > best_auc or (val_auc == best_auc and val_loss < best_loss):
                best_auc = val_auc
                best_loss = val_loss
                best_model = dpfm_model

        return best_model, histories
        
def PredictEvaluateModel(model, test_feature_index, test_feature_value, y_true, threshold):
    y_pred = model.predict([test_feature_index, test_feature_value])

    y_pred_binary = (y_pred >= threshold).astype(int)

    log_loss_value = log_loss(y_true, y_pred)
    auc_value = roc_auc_score(y_true, y_pred)
    accuracy_value = accuracy_score(y_true, y_pred_binary)
    precision_value = precision_score(y_true, y_pred_binary)
    recall_value = recall_score(y_true, y_pred_binary)
    f1_value = f1_score(y_true, y_pred_binary)

    print(f"Log Loss: {log_loss_value:.4f}")
    print(f"AUC Score: {auc_value:.4f}")
    print(f"Accuracy: {accuracy_value:.4f}")
    print(f"Precision: {precision_value:.4f}")
    print(f"Recall: {recall_value:.4f}")
    print(f"F1 Score: {f1_value:.4f}")

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color="red", label="Threshold Points")
    plt.title("ROC Curve with AUC")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

    return {
        "log loss": round(log_loss_value, 3),
        "roc auc": round(auc_value, 3),
        "precision": round(precision_value, 3),
        "accuracy": round(accuracy_value, 3),
        "recall": round(recall_value, 3),
        "f1 score": round(f1_value, 3)
    }

if __name__ == "__main__":
    print(f'This is Def Function Script')