import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, roc_auc_score, precision_score, recall_score, f1_score, roc_curve, accuracy_score
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
    test_y = evaluate_test['Predicted'].values

    dfm_params = {
        "embedding_size":8,
        "deep_layers":[32, 32],
        "dropout_deep":[0.5, 0.5],
        "dropout_fm":[1.0, 1.0],
        "batch_size":1024,
        "learning_rate":0.001,
        "optimizer":"adam",
        "loss":'binary_crossentropy',
        "metrics":["AUC"],
        "epochs":10
    }

    feature_size = total_feature
    field_size = feature_index.shape[1]

    deepfm_model = build_deepfm(
        feature_size=feature_size,
        field_size=field_size,
        params=dfm_params
    )

    deepfm_model.compile(
        optimizer=tf.keras.optimizers.get({
            "class_name": dfm_params['optimizer'],
            "config": {"learning_rate": dfm_params["learning_rate"]}
        }),
        loss=dfm_params["loss"],
        metrics=dfm_params["metrics"]
    )

    deepfm_model.fit(
        x=[feature_index, feature_value],
        y=train_y,
        batch_size=dfm_params["batch_size"],
        epochs=dfm_params["epochs"],
        validation_data=([evaluate_feature_index, evaluate_feature_value], test_y)
    )

    y_pred = deepfm_model.predict([evaluate_feature_index, evaluate_feature_value])
    y_pred_binary = (y_pred > 0.5).astype(int)

    auc = roc_auc_score(test_y, y_pred)
    logloss = log_loss(test_y, y_pred)
    accuracy = accuracy_score(test_y, y_pred_binary)
    precision = precision_score(test_y, y_pred_binary)
    recall = recall_score(test_y, y_pred_binary)
    f1 = f1_score(test_y, y_pred_binary)

    print(f"AUC Score: {auc:.3f}")
    print(f"Log Loss: {logloss:3f}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")

    fpr, tpr, thresholds = roc_curve(test_y, y_pred)
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

    deepfm_model.save("deepfm_model.h5")