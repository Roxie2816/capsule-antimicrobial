# bert_capsule_model.py
import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten, concatenate, Reshape,BatchNormalization, MultiHeadAttention, Concatenate, Dropout, Activation, Bidirectional, LSTM

from Capsule_MPNN import PrimaryCap, Capsule, Length

# from tensorflow.keras.utils.generic_utils import get_custom_objects
# get_custom_objects()['squash'] = squash

def custom_f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
        possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
        recall = (true_positives + tf.keras.backend.epsilon()) / (possible_positives + tf.keras.backend.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
        predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
        precision = (true_positives + tf.keras.backend.epsilon()) / (predicted_positives + tf.keras.backend.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

def split(drug, y, drug_encoder, name):
    if not os.path.exists(name):
        os.makedirs(name)

    if drug_encoder == "chemmolefusion":
        train_d, test_d, train_y, test_y = train_test_split(drug, y, test_size=0.2,
                                                                                random_state=500, stratify=y)  
        with open(name + "/sample_summary.txt", "w") as fw:
            fw.write("train_drug\ttest_drug\ttrain_y\ttest_y\n")
    train_y = to_categorical(train_y, num_classes=2)
    test_y = to_categorical(test_y, num_classes=2)
    return train_d, test_d, train_y, test_y


def ASL(gamma_neg: float = 4, 
        gamma_pos: float = 1,
        clip: float = 0.05, 
        from_logits: bool = False, 
        label_smoothing: float = 0.05,
        alpha: Union[float, str] = None,  # 允许float或'auto'
        neg_weight: float = 0.25) -> Callable:
    
    assert 0 <= neg_weight <= 1, "neg_weight should be in [0, 1]"
    if alpha is not None and alpha != 'auto':
        assert 0 <= alpha <= 1, "alpha should be in [0, 1] or 'auto'"

    def losses(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        nonlocal alpha 
        
        y_true = tf.cast(y_true, tf.float32)
        
        if label_smoothing > 0:
            y_true = y_true * (1 - label_smoothing) + 0.5 * label_smoothing
        
        if from_logits:
            y_pred = tf.sigmoid(y_pred)
        y_pred = tf.clip_by_value(y_pred, clip, 1 - clip)
        
        if alpha == 'auto':
            pos_ratio = tf.reduce_mean(y_true)
            current_alpha = 0.5 / (pos_ratio + tf.keras.backend.epsilon())
            current_alpha = tf.clip_by_value(current_alpha, 0.1, 10)
        else:
            current_alpha = alpha if alpha is not None else 0.5
        
        p_pos = y_pred
        p_neg = 1 - y_pred
        pt_pos = y_true * p_pos + (1 - y_true) * (1 - p_pos)
        pt_neg = y_true * p_neg + (1 - y_true) * (1 - p_neg)
        
        loss_pos = -tf.pow(1 - pt_pos, gamma_pos) * tf.math.log(pt_pos + tf.keras.backend.epsilon())
        loss_neg = -neg_weight * tf.pow(pt_neg, gamma_neg) * tf.math.log(1 - pt_neg + tf.keras.backend.epsilon())
        
        if current_alpha is not None:
            loss = current_alpha * loss_pos + (1 - current_alpha) * loss_neg
        else:
            loss = loss_pos + loss_neg
            
        return tf.reduce_mean(loss)
    
    return losses

def model_bert_chemmolefusion_capsule(param):
    sequence_input_2 = Input(shape=(1536,))
    model_d = Flatten()(sequence_input_2)
    model_d = Dense(param['target_dense'], kernel_regularizer=keras.regularizers.l2(0.01))(model_d)
    model_d = BatchNormalization()(model_d)
    model_d = Activation('relu')(model_d)

    model = Reshape((-1, 8))(model_d)

    primarycaps = PrimaryCap(model, dim_vector=8, n_channels=8, kernel_size=param['kernel_size'], strides=1,
                             padding='valid')
    capsule = Capsule(num_capsule=param['num_capsule'], dim_capsule=16, routings=param['routings'],
                      share_weights=True)(primarycaps)

    length = Length()(capsule)

    model = keras.Model(inputs=sequence_input_2, outputs=length)
    model.summary()

    model.compile(optimizer='adam',loss=focal_plus(gamma_pos=4.0, gamma_neg=1.0, clip=0.2),metrics=[custom_f1])
    
    return model

param = {
    'target_dense': 128,
    'kernel_size': 3,
    'num_capsule': 2,
    'routings': 3
}


model = model_bert_chemmolefusion_capsule(param)
