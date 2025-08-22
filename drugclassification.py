import argparse
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint, CSVLogger
from sklearn.metrics import accuracy_score, f1_score, auc, precision_recall_curve, roc_curve, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn import metrics
from sklearn.metrics import make_scorer, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, \
    average_precision_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from keras.callbacks import LearningRateScheduler
from keras.callbacks import EarlyStopping
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import pickle

from bert_capsule_model import *
from Capsule_MPNN import *
# from ChemBERTa import *

# from ChemBERTa_mole_fusion import *
from Mole_Bert import *
# from bert import *
# from ReadData import *
from feature_extraction import get_feature

import tensorflow.keras.backend as K
# def custom_f1(y_true, y_pred):
#     def recall(y_true, y_pred):
#         true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#         possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#         recall = (true_positives + K.epsilon()) / (possible_positives + K.epsilon())
#         return recall

#     def precision(y_true, y_pred):
#         true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#         predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#         precision = (true_positives + K.epsilon()) / (predicted_positives + K.epsilon())
#         return precision

#     precision = precision(y_true, y_pred)
#     recall = recall(y_true, y_pred)
#     return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def custom_f1(y_true, y_pred):
    def recall(y_true, y_pred):
        # 将 y_true 转换为 float32 类型
        y_true = K.cast(y_true, 'float32')
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = (true_positives + K.epsilon()) / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        # 将 y_true 转换为 float32 类型
        y_true = K.cast(y_true, 'float32')
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = (true_positives + K.epsilon()) / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

import math
def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.5
    epochs_drop = 50.0
    lrate = initial_lrate * math.pow(drop,
                                     math.floor((1 + epoch) / epochs_drop))
    return lrate

# def evaluate(model_parh, test_d, test_y, model_name):
def evaluate(model_path_prefix, test_d, test_y, model_name):
    ###
    # model_files = [f for f in os.listdir(model_path_prefix) if f.endswith('.h5')]
    # model_files.sort()
    # if model_files:
    #     last_model_file = os.path.join(model_path_prefix, model_files[-1])
    # else:
    #     print("未找到 .h5 模型文件。")
    #     return
    
    # print(f"尝试加载的模型路径: {last_model_file}", flush=True)
    ###
    # print(f"尝试加载的模型路径: {model_parh}", flush=True)
    print("=============Start Evaluate! ===========", flush=True)
    # model = load_model(last_model_file, compile=False, ###
    model = load_model(model_path_prefix, compile=False,
                           custom_objects={'Capsule': Capsule, 'Length': Length,'TransformerEncoderReadout': TransformerEncoderReadout, 'squash': squash}) ###新增了一个squash
    pred0 = model.predict([np.array(test_d)]) 
    # loss, _, accuracy, auc, precision, recall, tp, tn, fp, fn = model.evaluate(np.array(test_d), np.array(test_y), verbose=0)
    pred = np.argmax(pred0, -1)
    confusion = metrics.confusion_matrix(np.array(test_y)[:, 1], pred)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]

    sensitivity = recall_score(np.array(test_y)[:, 1], pred)
    specificity = TN / float(TN + FP)
    precision = precision_score(np.array(test_y)[:, 1], pred)
    accuracy = accuracy_score(np.array(test_y)[:, 1], pred)
    f1 = f1_score(np.array(test_y)[:, 1], pred)
    aucroc = roc_auc_score(np.array(test_y)[:, 1], pred0[:, 1])
    # aupr=average_precision_score(np.array(test_y)[:,1],pred0[:,1])
    fpr0, tpr0, thresholds0 = metrics.roc_curve(np.array(test_y)[:, 1], pred0[:, 1])
    auc_v = metrics.auc(fpr0, tpr0)
    precision0, recall, thresholds = metrics.precision_recall_curve(np.array(test_y)[:, 1], pred0[:, 1])
    area = metrics.auc(recall, precision0)
    print("sensitivity", round(sensitivity, 3))
    print("specificity", round(specificity, 3))
    print("precision", round(precision, 3))
    print("accuracy", round(accuracy, 3))
    print("f1", round(f1, 3))
    print("TP:", TP)
    print("TN:", TN)
    print("FP:", FP)
    print("FN:", FN)
    print("aucroc:", round(auc_v, 3))
    print("aupr:", round(area, 3))
    # print("Loss:", loss)

    with open(os.path.join(model_name, "performance.txt"), "w") as fw:
        fw.write(
            "Evaluation metrics" + "\t" + "sensitivity" + "\t" + "specificity" + "\t" + "precision" + "\t" + "accuracy" + "\t" + "f1" + "\t" + "aucroc" + "\t" + "aupr" + "\n")
        fw.write("Value" + "\t" + str(round(sensitivity, 3)) + "\t" + str(round(specificity, 3)) + "\t" + str(
            round(precision, 3)) + "\t" + str(round(accuracy, 3)) + "\t" + str(round(f1, 3)) + "\t" + str(
            round(auc_v, 3)) + "\t" + str(round(area, 3)) + "\n")
        # df_evaluate = pd.DataFrame({"dataset":[dti],"classifier": [model_name], "accuracy": [
        df_evaluate = pd.DataFrame({"classifier": [model_name], "accuracy": [str(accuracy)], "specificity": [str(specificity)],"sensitivity": [str(sensitivity)], "aucroc": [str(auc_v)], "aupr": [str(area)],"f1": [str(f1)]})
        df_evaluate.to_csv("./output/evaluate-performance.csv", mode="a", index=None)
        # df_evaluate.to_csv("./result/evaluate-performance_2.csv", mode="a", index=None)
    print("=============Evaluate Over! ===========", flush=True)
    return sensitivity, specificity, precision, accuracy, f1, auc_v, area


def plot_training_history(history, name):
    epochs = range(len(history.history['accuracy']))
    plt.figure(dpi=300)
    plt.plot(epochs, history.history['accuracy'], '#228B8B', label='Training acc')
    plt.scatter(epochs, history.history['accuracy'], color='#228B8B', s=12)
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('./' + name + "/" + name + '_acc.jpg')

    plt.figure(dpi=300)
    plt.plot(epochs, history.history['loss'], '#228B8B', label='Training loss')
    plt.scatter(epochs, history.history['loss'], color='#228B8B', s=12)
    # plt.plot(epochs,history.history['val_loss'],'#8B2222',label='Validation val_loss')
    # plt.scatter(epochs,history.history['val_loss'],color='#8B2222',s=12)
    plt.title('Training and Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./' + name + "/" + name + '_loss.jpg')

def fitting(train_d, test_d, train_y, test_y, model_type, lr, ep, path, batchsize=32):
    parameters_string = model_type
    train_path = os.path.join(path, parameters_string)
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    fw = open(model_type + '/test_d.txt', 'wb')
    pickle.dump(test_d, fw)
    fw.close()
    fw = open(model_type + '/test_y.txt', 'wb')
    pickle.dump(test_y, fw)
    fw.close()

    cv = KFold(n_splits=5, random_state=33, shuffle=True)

    if "bert_chemmolefusion_capsule" in model_type:
        param_grid = {
            "drug_dense": [200, 400],
            "batch_size": [32],
            "message_units": [64],
            "message_steps": [4],
            "num_attention_heads": [8],
            "dense_units": [512],
            "num_capsule": [2],
            "routings": [3, 6],
            "kernel_size": [5, 10],
            "target_dense": [200],
        }
        fw = open(os.path.join(train_path, "search_process.txt"), "w")
        fw.close()
        best_score, best_drug_dense, best_batch_size, best_message_units, best_message_steps, best_num_attention_heads, best_dense_units, best_num_capsule, best_routings, best_kernel_size = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        for drug_dense in param_grid["drug_dense"]:
            for batch_size in param_grid["batch_size"]:
                for message_units in param_grid["message_units"]:
                    for message_steps in param_grid["message_steps"]:
                        for num_attention_heads in param_grid["num_attention_heads"]:
                            for dense_units in param_grid["dense_units"]:
                                for num_capsule in param_grid["num_capsule"]:
                                    for routings in param_grid["routings"]:
                                        for kernel_size in param_grid["kernel_size"]:
                                            all_acc_scores = []
                                            all_sensitivity = []
                                            all_specificity = []
                                            all_aucroc = []
                                            all_aupr = []
                                            all_f1 = []
                                            fw = open(os.path.join(train_path, "search_process.txt"), "a")
                                            fw.write(
                                                "drug_dense,batch_size,message_units,message_steps,num_attention_heads,dense_units,num_capsule,routings,kernel_size, target_dense\n")
                                            fw.write(str(drug_dense) + "," + str(batch_size) + "," + str(
                                                message_units) + "," + str(message_steps) + "," + str(
                                                num_attention_heads) + "," + str(dense_units) + "," + str(
                                                num_capsule) + "," + str(routings) + "," + str(kernel_size) + "\n")
                                            fw.close()
                                            for i, (train_index, val_index) in enumerate(cv.split(train_d)):
                                                train_d_train, train_d_val = np.array(train_d)[train_index], np.array(train_d)[val_index]
                                                train_y_train, train_y_val = train_y[train_index], train_y[val_index]
                                                print("train_d_train shape:", train_d_train.shape) ###
                                                print("train_y_train shape:", train_y_train.shape) ###
                                                param = {
                                                    "drug_dense": drug_dense,
                                                    "target_dense": drug_dense,
                                                    "kernel_size": kernel_size,
                                                    "num_capsule": num_capsule, "routings": routings}
                                                model_1 = model_bert_chemmolefusion_capsule(param=param)
                                                adam = Adam(learning_rate=lr)
                                                # model_1.compile(loss=focal_plus(gamma_pos=4.0, gamma_neg=1.0, clip=0.2), optimizer=adam, ###
                                                # model_1.compile(loss=BalancedBCE(pos_weight=None, label_smoothing=0.05,reduction='mean'), optimizer=adam,
                                                model_1.compile(loss=ASL(gamma_neg=4, gamma_pos=1.5, neg_weight=0.2, alpha='auto', clip=0.05), optimizer=adam, #调参
                                                                
                                                                metrics=[custom_f1, 'accuracy', 'AUC',
                                                                         tf.keras.metrics.Precision(),
                                                                         tf.keras.metrics.Recall(),
                                                                         tf.keras.metrics.TruePositives(),
                                                                         tf.keras.metrics.TrueNegatives(),
                                                                         tf.keras.metrics.FalsePositives(),
                                                                         tf.keras.metrics.FalseNegatives()])
                                                lrate = LearningRateScheduler(step_decay)
                                                Early = EarlyStopping(monitor="accuracy", mode='max', patience=50, ###更改早停
                                                                      verbose=1, restore_best_weights=True)
                                                model_1.fit(train_d_train, train_y_train, epochs=ep,
                                                            batch_size=batchsize,
                                                            verbose=2,
                                                            callbacks=[lrate, Early])
                                                # pred0 = model_1.predict([train_m,train_m])
                                                pred0 = model_1.predict(train_d_val)
                                                pred = np.argmax(pred0, -1)
                                                confusion = metrics.confusion_matrix(np.array(train_y_val)[:, 1], pred)
                                                TN = confusion[0, 0]
                                                FP = confusion[0, 1]
                                                accuracy = accuracy_score(np.array(train_y_val)[:, 1], pred)
                                                specificity = TN / float(TN + FP)

                                                fpr0, tpr0, thresholds0 = metrics.roc_curve(np.array(train_y_val)[:, 1],
                                                                                            pred)
                                                f1 = f1_score(np.array(train_y_val)[:, 1], pred)
                                                auc_v = metrics.auc(fpr0, tpr0)
                                                precision0, recall, thresholds = metrics.precision_recall_curve(
                                                    np.array(train_y_val)[:, 1], pred)
                                                area = metrics.auc(recall, precision0)
                                                recall = recall_score(np.array(train_y_val)[:, 1], pred)
                                                fw = open(os.path.join(train_path, "search_process.txt"), "a")
                                                # fw.write("#####Fold" + str(i) + "\n")
                                                fw.write("accuracy:" + str(round(accuracy, 4)) + "\t")
                                                fw.write("specificity:" + str(round(specificity, 4)) + "\t")
                                                fw.write("sensitivity:" + str(round(recall, 4)) + "\t")
                                                fw.write("aucroc:" + str(round(auc_v, 4)) + "\t")
                                                fw.write("aupr:" + str(round(area, 4)) + "\t")
                                                fw.write("f1:" + str(round(f1, 4)) + "\n")
                                                all_acc_scores.append(accuracy)
                                                all_f1.append(f1)
                                                all_aupr.append(area)
                                                all_aucroc.append(auc_v)
                                                all_specificity.append(specificity)
                                                all_sensitivity.append(recall)
                                                fw.close()

                                            score = np.mean(all_acc_scores)
                                            if score > best_score:
                                                fw = open(os.path.join(train_path, "search_process.txt"), "a")
                                                fw.write("accuracy:" + str(score) + " > best_score:" + str(
                                                    best_score) + "\n")
                                                fw.close()
                                                best_score, best_drug_dense, best_batch_size, best_message_units, best_message_steps, best_num_attention_heads, best_dense_units, best_num_capsule, best_routings, best_kernel_size = score, drug_dense, batch_size, message_units, message_steps, num_attention_heads, dense_units, num_capsule, routings, kernel_size
        fw = open(os.path.join(train_path, "search_process.txt"), "a")
        fw.write(
            "best_drug_dense,best_batch_size,best_message_units,best_message_steps,best_num_attention_heads,best_dense_units,best_num_capsule,best_routings,best_kernel_size\n")
        fw.write(str(best_drug_dense) + "," + str(best_batch_size) + "," + str(
            best_message_units) + "," + str(best_message_steps) + "," + str(
            best_num_attention_heads) + "," + str(
            best_dense_units) + "," + str(best_num_capsule) + "," + str(
            best_routings) + "," + str(
            best_kernel_size) + "\n")
        fw.write(
            "accuracy,specificity,sensitivity,aucroc,aupr,f1\n")
        fw.write(str(best_score) + "±" + str(np.std(all_acc_scores)) + "," + str(
            np.mean(all_specificity)) + "±" + str(np.std(all_specificity)) + str(
            np.mean(all_sensitivity)) + "±" + str(np.std(all_sensitivity)) + str(
            np.mean(all_aucroc)) + "±" + str(np.std(all_aucroc)) + str(
            np.mean(all_aupr)) + "±" + str(np.std(all_aupr)) + str(
            np.mean(all_f1)) + "±" + str(np.std(all_f1)) + "\n")
        # df = pd.DataFrame({"dataset": [dti], "classifier": [model_type], "accuracy": [
        df = pd.DataFrame({"classifier": [model_type], "accuracy": [
            str(best_score) + "±" + str(np.std(all_acc_scores))], "specificity": [
            str(np.mean(all_specificity)) + "±" + str(np.std(all_specificity))],
                           "sensitivity": [str(np.mean(all_sensitivity)) + "±" + str(
                               np.std(all_sensitivity))], "aucroc": [
                str(np.mean(all_aucroc)) + "±" + str(np.std(all_aucroc))], "aupr": [
                str(np.mean(all_aupr)) + "±" + str(np.std(all_aupr))],
                           "f1": [str(np.mean(all_f1)) + "±" + str(np.std(all_f1))]})
        df.to_csv("./output/5fold-performance.csv", mode="a", index=None) ###
        # df.to_csv("./result/5fold-performance_2.csv", mode="a", index=None) ###
        fw.close()
        adam = Adam(learning_rate=lr)
        param = {"drug_dense": drug_dense,
                 "kernel_size": 5, "num_capsule": 2, "routings": 3, "target_dense": 200}
        model_1 = model_bert_chemmolefusion_capsule(param=param)
        # model_1.compile(loss=focal_plus(gamma_pos=4.0, gamma_neg=1.0, clip=0.2), optimizer=adam,
        # model_1.compile(loss=BalancedBCE(pos_weight=None, label_smoothing=0.05,reduction='mean'), optimizer=adam,
        model_1.compile(loss=ASL(gamma_neg=4, gamma_pos=1.5, neg_weight=0.2, alpha='auto', clip=0.05), optimizer=adam, #调参
                        metrics=[custom_f1, 'accuracy', 'AUC', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(),
                                 tf.keras.metrics.TruePositives(), tf.keras.metrics.TrueNegatives(),
                                 tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives()])
        # dti_name = dti.split('/')
        save_dir = os.path.join(train_path, 'checkpoints')
        # model_parh = os.path.join(train_path + '_', "{}.ckpt")
        model_parh = os.path.join(train_path, "model_epoch_{epoch:02d}.ckpt")
        csv_logger = CSVLogger(os.path.join(train_path, 'model_training.csv'))
        # csv_logger = CSVLogger(os.path.join(train_path, 'model_training_2.csv'))
        lrate = LearningRateScheduler(step_decay)
        Early = EarlyStopping(monitor='accuracy', mode='max', patience=50, verbose=1) #更改早停
        # correct_filepath = os.path.splitext(model_parh)[0] + '.keras'
        # correct_filepath = os.path.splitext(model_parh)[0] + '.h5'
        correct_filepath = os.path.join(save_dir, 'best_model.h5')
        checkpoint = ModelCheckpoint(filepath=correct_filepath, monitor='accuracy', mode='max', save_best_only=True,
                                     verbose=1) ###
        history = model_1.fit(train_d, train_y, batch_size=batchsize, epochs=ep,
                              callbacks=[lrate, Early, csv_logger, checkpoint], verbose=0)
        print("=============Train Over! ===========", flush=True)

    return history, model_1
    # return model_1

    # 确保输出目录存在
# def main():
#     import traceback
#     parser = argparse.ArgumentParser(description='Drug Classification')
#     parser.add_argument('--input-path', type=str, required=True, help='Data input path')
#     parser.add_argument('--model-name', type=str, required=True, help='Model name')
#     parser.add_argument('--drug-descriptor', type=str, required=True, 
#                        choices=['fusion', 'chembert', 'molebert', 'smolebert', 'molclr', 'molformer'],
#                        help='Drug Descriptor')
    
#     parser.add_argument('--train', action='store_true', help='Training model or not')
#     parser.add_argument('--model-dir', type=str, default='output', help='Saved model path')
#     parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate for the model')
#     parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
#     parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs')
#     parser.add_argument('--output-dir', type=str, default='data', help='Output directory for features')
#     parser.add_argument('-g', '--gpu', type=int, default=0, help='GPU ID')
#     parser.add_argument('-sl', '--sequence-length', type=int, default=1024, help='Sequence length')
#     parser.add_argument('--taxonomy', type=str, help='Taxonomy information')
#     args = parser.parse_args()

#     drug_descriptor = args.drug_descriptor
#     model_name = args.model_name
#     learning_rate = args.learning_rate
#     n_epoch = args.epochs
#     batch_size = args.batch_size
#     # 创建模型目录
#     if not os.path.exists(args.model_name):
#         os.makedirs(args.model_name)
    
#     try:
#         # 1. 特征提取
#         feature_df = get_feature(
#             input_path=args.input_path,
#             drug_descriptor=args.drug_descriptor
#         )
        
#         # 保存特征
#         feature_path = os.path.join(args.output_dir, f'{args.drug_descriptor}_features.csv')
#         feature_df.to_csv(feature_path, index=False)
#         print(f"特征已保存到: {feature_path}")
        
#         # 2. 模型训练（如果指定了--train）
#         if args.train:
#             if 'Activity' not in feature_df.columns:
#                 raise ValueError("训练需要Activity列作为标签")
#             print("\n开始模型训练...")
#             drug = feature_df.drop('Activity', axis=1).values
#             y = feature_df['Activity'].tolist()
#             train_d, test_d, train_y, test_y = train_test_split(drug, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None)
#             history1, model = fitting(train_d, test_d, train_y, test_y, model_name, learning_rate, n_epoch, args.model_dir, batch_size)
            
#             plot_training_history(history1, args.model_name)
#             print("训练完成！")
            
#     except Exception as e:
#         print(f"处理失败: {str(e)}")
#         traceback.format_exc()
#         exit(1)
#         # 这里可以添加模型训练代码
#         # train_model(feature_df, args)

def main():
    import traceback
    parser = argparse.ArgumentParser(description='Drug Classification')
    parser.add_argument('--input-path', type=str, help='Path to dataset')
    parser.add_argument('--drug-descripter', type=str, required=True, help='Drug descriptor')

    parser.add_argument('--train', action='store_true', help='Train Model option')
    parser.add_argument('--predict', action='store_true', help='Predict external data option')
    parser.add_argument('--predict-csv', type=str,
                        help='CSV file to predict (required if --predict)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the model')
    # parser.add_argument('--data_prefix', type=str, help='Data prefix')
    parser.add_argument('--model-name', type=str, required=True, help='Model name')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('-dp', '--data-path', type=str, default='data', help='Data path')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('-sl', '--sequence-length', type=int, default=1024, help='Sequence length')
    parser.add_argument('--taxonomy', type=str, help='Taxonomy information') ### required=true
    args = parser.parse_args()

    # input_path = args.input_path
    drug_descripter = args.drug_descripter
    model_name = args.model_name
    learning_rate = args.learning_rate
    n_epoch = args.epochs
    batch_size = args.batch_size
    # data_prefix = args.data_prefix

    if not os.path.exists(model_name):
        os.makedirs(model_name)

    try:
        # 1. 特征提取
        feature_df = get_feature(
            input_path=args.input_path,
            drug_descripter=args.drug_descripter
        )
        
        # 保存特征
        feature_path = os.path.join('./output', f'{args.drug_descripter}_features.csv')
        feature_df.to_csv(feature_path, index=False)
        print(f"特征已保存到: {feature_path}")
        
        # 2. 模型训练（如果指定了--train）
        if args.train:
            if 'Activity' not in feature_df.columns:
                raise ValueError("训练需要Activity列作为标签")
            print("\n开始模型训练...")
            drug = feature_df.drop('Activity', axis=1).values
            y = feature_df['Activity'].values
            y = to_categorical(y)
            # drug = feature_df.iloc[:, feature_df.columns != 'Activity'].values
            # y = feature_df.loc[:, 'Activity']
            train_d, test_d, train_y, test_y = train_test_split(drug, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None)
            history1, model = fitting(train_d, test_d, train_y, test_y, model_name, learning_rate, n_epoch, model_name, batch_size)
            # evaluate("./" + model_name +"_" + "/" + model_name  +".h5", test_d, test_y, model_name)
            train_path = os.path.join(model_name, model_name)
            best_model_path = os.path.join(train_path, 'checkpoints', 'best_model.h5')
            evaluate(best_model_path, test_d, test_y, args.model_name)
            plot_training_history(history1, args.model_name)

        elif args.predict:
            # if not args.predict_csv:
                # raise ValueError('--predict 时必须提供 --predict-csv')

            # 1. 拼模型路径
            train_path = os.path.join(args.model_name, args.model_name)
            best_model_path = os.path.join(train_path, 'checkpoints', 'best_model.h5')
            model = keras.models.load_model(best_model_path, compile=False, custom_objects={'Capsule': Capsule, 'Length': Length,'TransformerEncoderReadout': TransformerEncoderReadout, 'squash': squash})
            print("Load model successfully...")

            # pred_feature_df = get_feature(
            #     input_path=args.predict_csv,          # 注意这里传的是外部 CSV
            #     drug_descripter=args.drug_descripter
            # )
            # 2. 读待预测的 CSV
            # df = pd.read_csv(args.predict_csv)          # 通过 argparse 传进来的文件路径
            # predict_d = pred_feature_df.drop(columns=['Activity'], errors='ignore').values   # 去掉 activity 列（如果有）
            # predict_y = pred_feature_df['Activity'] if 'Activity' in pred_feature_df.columns else None
            predict_d = feature_df.drop(columns=['Activity'], errors='ignore').values   # 去掉 activity 列（如果有）
            predict_y = feature_df['Activity'] if 'Activity' in feature_df.columns else None

            # 3. 预测
            proba = model.predict([np.array(predict_d)])     # 返回 [[p_true, p_false], ...]
            pred_label = (proba[:, 1] > 0.5).astype(int)
            true_proba  = proba[:, 1]
            false_proba = proba[:, 0]

            # 4. 根据是否有 ground-truth 决定输出
            if predict_y is not None:
                evaluate(best_model_path, predict_d, predict_y, args.model_name)
            else:
                predict_results = pd.DataFrame({
                    'predicted_label': pred_label,
                    'true_probability': true_proba,
                    'false_probability': false_proba
                })
                predict_results.to_csv("./output/prediction-result.csv", mode="a", index=None)
                print(predict_results)              
            
        
            
    except Exception as e:
        print(f"处理失败: {str(e)}")
        traceback.format_exc()
        exit(1)
        # 这里可以添加模型训练代码
        # train_model(feature_df, args)
if __name__ == "__main__":
    main()
