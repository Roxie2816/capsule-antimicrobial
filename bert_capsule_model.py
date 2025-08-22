# bert_capsule_model.py
import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
# from src.loss_functions.losses import AsymmetricLoss ###
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
                                                                                random_state=500, stratify=y)  # 种子：1,42,500,800
        with open(name + "/sample_summary.txt", "w") as fw:
            fw.write("train_drug\ttest_drug\ttrain_y\ttest_y\n")
    train_y = to_categorical(train_y, num_classes=2)
    test_y = to_categorical(test_y, num_classes=2)
    return train_d, test_d, train_y, test_y
    # return train_d, test_d, np.array(train_y), np.array(test_y)

def AsymmetricLoss(gamma_pos=0, gamma_neg=4, clip=0.2):
    def losses(y_true, y_pred):
        eps = 1e-8
        y_pred = keras.backend.sigmoid(y_pred)
        xs_pos = y_pred
        xs_neg = 1 - y_pred

        if clip is not None and clip > 0:
            xs_neg = keras.backend.clip(xs_neg + clip, 0, 1)

        los_pos = y_true * keras.backend.log(keras.backend.clip(xs_pos, eps, 1))
        los_neg = (1 - y_true) * keras.backend.log(keras.backend.clip(xs_neg, eps, 1))
        loss = los_pos + los_neg

        if gamma_neg > 0 or gamma_pos > 0:
            pt0 = xs_pos * y_true
            pt1 = xs_neg * (1 - y_true)
            pt = pt0 + pt1
            one_sided_gamma = gamma_pos * y_true + gamma_neg * (1 - y_true)
            one_sided_w = keras.backend.pow(1 - pt, one_sided_gamma)
            loss *= one_sided_w

        return -keras.backend.sum(loss)
    return losses

def ASLSingleLabel(gamma_pos=0, gamma_neg=10, eps=0.05, reduction='mean'):
    def losses(y_true, y_pred):
        num_classes = keras.backend.int_shape(y_pred)[-1]

        log_preds = keras.backend.log(y_pred + keras.backend.epsilon())

        if keras.backend.ndim(y_true) == 1:
            y_true = keras.backend.one_hot(keras.backend.cast(y_true, 'int32'), num_classes)

        anti_targets = 1 - y_true
        xs_pos = keras.backend.exp(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * y_true
        xs_neg = xs_neg * anti_targets
        asymmetric_w = keras.backend.pow(1 - xs_pos - xs_neg,
                            gamma_pos * y_true + gamma_neg * anti_targets)
        log_preds = log_preds * asymmetric_w

        if eps > 0: 
            y_true = y_true * (1 - eps) + (eps / num_classes)

        loss = - y_true * log_preds
        loss = keras.backend.sum(loss, axis=-1)

        if reduction =='mean':
            loss = keras.backend.mean(loss)
        elif reduction =='sum':
            loss = keras.backend.sum(loss)

        return loss
    return losses


def BinaryImbalanceLoss(gamma_pos=0, gamma_neg=4, 
                       eps=0.1, alpha=0.25, 
                       label_smoothing=0.05,
                       pos_weight=None,
                       focal_weight=0.7,
                       reduction='mean'):
    """
    专为二分类设计的增强版不平衡损失函数
    
    参数:
        gamma_pos: 正样本的焦点参数(通常设为0或较小值)
        gamma_neg: 负样本的焦点参数(较大值以抑制简单负样本)
        eps: 非对称概率偏移量
        alpha: 正样本的权重平衡因子
        label_smoothing: 标签平滑系数
        pos_weight: 正样本的权重(建议设为负样本数/正样本数)
        focal_weight: 焦点损失成分的权重(0-1)
        reduction: 'mean'或'sum'指定损失缩减方式
    """
    
    def loss_function(y_true, y_pred):
        # 确保y_true是float32类型
        y_true = tf.cast(y_true, tf.float32)
        
        # 应用标签平滑
        if label_smoothing > 0:
            y_true = y_true * (1 - label_smoothing) + label_smoothing * 0.5
        
        # 计算概率和log概率
        y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        log_preds = tf.keras.backend.log(y_pred)
        log_preds_neg = tf.keras.backend.log(1 - y_pred)
        
        # 非对称焦点权重计算
        pos_loss = -tf.keras.backend.pow(1 - y_pred, gamma_pos) * log_preds * y_true
        neg_loss = -tf.keras.backend.pow(y_pred, gamma_neg) * log_preds_neg * (1 - y_true)
        
        # 组合基础CE和焦点损失
        ce_loss = -y_true * log_preds - (1 - y_true) * log_preds_neg
        focal_loss = pos_loss + neg_loss
        loss = (focal_weight * focal_loss) + ((1 - focal_weight) * ce_loss)
        
        # 应用alpha平衡
        if alpha is not None:
            alpha_weight = y_true * alpha + (1 - y_true) * (1 - alpha)
            loss = loss * alpha_weight
        
        # 应用正样本权重
        if pos_weight is not None:
            pos_weight_tensor = y_true * pos_weight + (1 - y_true)
            loss = loss * pos_weight_tensor
        
        # 应用概率偏移
        if eps > 0:
            offset = log_preds_neg * eps
            loss = loss + offset
        
        # 应用缩减
        if reduction == 'mean':
            loss = tf.keras.backend.mean(loss)
        elif reduction == 'sum':
            loss = tf.keras.backend.sum(loss)

        if tf.executing_eagerly():
            y_pred_class = tf.cast(y_pred > 0.5, tf.float32)
            tp = tf.reduce_sum(y_true * y_pred_class)
            fn = tf.reduce_sum(y_true * (1 - y_pred_class))
            sensitivity = tp / (tp + fn + tf.keras.backend.epsilon())
            loss = loss - 0.3 * sensitivity  # 0.3是敏感度优化强度
        
        return loss
    
    return loss_function


def focal_plus(gamma_pos=4.0, gamma_neg=1.0, clip=0.2):
    def loss(y_true, y_pred):
        # 保证y_true和y_pred是float32
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-8, 1.0 - 1e-8)   # 防止log(0)

        # 正样本预测概率 p_t = p
        pos_p_t = y_pred
        # 负样本预测概率 p_t = 1 - p
        neg_p_t = 1.0 - y_pred

        # 焦点因子
        pos_factor = tf.pow(1.0 - pos_p_t, gamma_pos)
        neg_factor = tf.pow(neg_p_t, gamma_neg)

        # 正负样本损失
        pos_loss = - y_true * pos_factor * tf.math.log(pos_p_t)
        neg_loss = - (1.0 - y_true) * neg_factor * tf.math.log(neg_p_t)

        # 负样本损失clip阈值处理
        if clip is not None and clip > 0:
            neg_loss = tf.maximum(neg_loss - clip, 0.0)

        # 总损失
        loss = pos_loss + neg_loss

        return tf.reduce_mean(loss)
    return loss

def BalancedBCE(alpha=0.5, pos_weight=None, label_smoothing=0.0, reduction='mean'):
    """
    稳定的带权重平衡二分类交叉熵损失函数 (基于 logits 输入)

    参数:
        alpha: float (0~1) - 用于正负样本加权，若 pos_weight 未设置，则使用 alpha
        pos_weight: float or None - 显式正样本权重，优先级高于 alpha，通常设为负样本数/正样本数
        label_smoothing: float (0~1) - 标签平滑系数
        reduction: str - 'mean'、'sum' 或 'none'，指定损失缩减方式

    返回:
        loss 函数，参数为 (y_true, y_pred_logits)
    """

    def loss(y_true, y_pred_logits):
        y_true = tf.cast(y_true, tf.float32)

        # 标签平滑
        if label_smoothing > 0:
            y_true = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing

        # 计算权重
        if pos_weight is not None:
            pw = tf.constant(pos_weight, dtype=tf.float32)
        else:
            # 将 alpha 映射到 pos_weight，公式为 pos_weight = alpha / (1 - alpha)
            # 但要避免除以0
            alpha_clipped = tf.clip_by_value(alpha, 1e-6, 1 - 1e-6)
            pw = alpha_clipped / (1.0 - alpha_clipped)

        # 使用 TensorFlow 数值稳定的加权交叉熵函数
        bce = tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=y_pred_logits, pos_weight=pw)

        # bce 的形状与 y_true 一致，进行缩减
        if reduction == 'mean':
            return tf.reduce_mean(bce)
        elif reduction == 'sum':
            return tf.reduce_sum(bce)
        elif reduction == 'none':
            return bce
        else:
            raise ValueError(f"Unsupported reduction mode: {reduction}")

    return loss


import tensorflow as tf
from typing import Optional, Callable, Union

def ASL(gamma_neg: float = 4, 
        gamma_pos: float = 1,
        clip: float = 0.05, 
        from_logits: bool = False, 
        label_smoothing: float = 0.05,
        alpha: Union[float, str] = None,  # 允许float或'auto'
        neg_weight: float = 0.25) -> Callable:
    
    # 参数验证
    assert 0 <= neg_weight <= 1, "neg_weight should be in [0, 1]"
    if alpha is not None and alpha != 'auto':
        assert 0 <= alpha <= 1, "alpha should be in [0, 1] or 'auto'"

    def losses(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        nonlocal alpha  # 声明alpha为非局部变量
        
        y_true = tf.cast(y_true, tf.float32)
        
        # 标签平滑
        if label_smoothing > 0:
            y_true = y_true * (1 - label_smoothing) + 0.5 * label_smoothing
        
        # 概率转换
        if from_logits:
            y_pred = tf.sigmoid(y_pred)
        y_pred = tf.clip_by_value(y_pred, clip, 1 - clip)
        
        # 自动平衡因子计算
        if alpha == 'auto':
            pos_ratio = tf.reduce_mean(y_true)
            current_alpha = 0.5 / (pos_ratio + tf.keras.backend.epsilon())
            current_alpha = tf.clip_by_value(current_alpha, 0.1, 10)
        else:
            current_alpha = alpha if alpha is not None else 0.5
        
        # 核心计算
        p_pos = y_pred
        p_neg = 1 - y_pred
        
        pt_pos = y_true * p_pos + (1 - y_true) * (1 - p_pos)
        pt_neg = y_true * p_neg + (1 - y_true) * (1 - p_neg)
        
        # 引入负样本权重和自动平衡
        loss_pos = -tf.pow(1 - pt_pos, gamma_pos) * tf.math.log(pt_pos + tf.keras.backend.epsilon())
        loss_neg = -neg_weight * tf.pow(pt_neg, gamma_neg) * tf.math.log(1 - pt_neg + tf.keras.backend.epsilon())
        
        if current_alpha is not None:
            loss = current_alpha * loss_pos + (1 - current_alpha) * loss_neg
        else:
            loss = loss_pos + loss_neg
            
        return tf.reduce_mean(loss)
    
    return losses

def model_bert_chemmolefusion_capsule(param):
    # 定义第二个输入，形状为 (2220,)
    sequence_input_2 = Input(shape=(1536,))
    model_d = Flatten()(sequence_input_2)
    model_d = Dense(param['target_dense'], kernel_regularizer=keras.regularizers.l2(0.01))(model_d)
    model_d = BatchNormalization()(model_d)
    model_d = Activation('relu')(model_d)

    # 融合两个输入的特征
    model = Reshape((-1, 8))(model_d)

    # 构建胶囊网络部分
    primarycaps = PrimaryCap(model, dim_vector=8, n_channels=8, kernel_size=param['kernel_size'], strides=1,
                             padding='valid')
    capsule = Capsule(num_capsule=param['num_capsule'], dim_capsule=16, routings=param['routings'],
                      share_weights=True)(primarycaps)

    # 计算胶囊网络输出的长度
    length = Length()(capsule)

    # 构建最终的模型
    model = keras.Model(inputs=sequence_input_2, outputs=length)
    model.summary()

    # 编译模型
    # model.compile(optimizer='adam', loss=focal_plus, metrics=[custom_f1])
    # model.compile(optimizer='adam', loss=AsymmetricLoss, metrics=[custom_f1])
    # model.compile(optimizer='adam', loss=ASLSingleLabel(), metrics=[custom_f1])
 
    # model.compile(optimizer='adam',loss=ASL(gamma_neg=4,gamma_pos=1.5,neg_weight=0.2,alpha='auto',clip=0.05),metrics=[custom_f1])

    # model.compile(optimizer='adam',loss=BalancedBCE(pos_weight=None,alpha=0.5,label_smoothing=0.05,reduction='mean'),metrics=[custom_f1])

    model.compile(optimizer='adam',loss=focal_plus(gamma_pos=4.0, gamma_neg=1.0, clip=0.2),metrics=[custom_f1])
    
    return model

# 示例参数
param = {
    'target_dense': 128,
    'kernel_size': 3,
    'num_capsule': 2,
    'routings': 3
}

# 创建模型
model = model_bert_chemmolefusion_capsule(param)