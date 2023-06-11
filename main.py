import json
import torch
# import os
from Manage.data import DataManage
from Manage.task import TaskManage
from Manage.model import *
from Manage.evaluate import *
from Utils.record import *
import warnings
import gc
warnings.filterwarnings("ignore")
 
CONTROL = [False ,False,True,False]  # 数据集控制位，True表示计算该数据集，False反之 （用于多服务器训练）
NUMCLASSES = 2
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('DEVICE: ', DEVICE)

# ModelName =  ["MSOA", "EEGNet", "DeepConvNet", "EEGInception", "PLNet", "HDCA", "xDAWNRG", "rLDA", "FDN"]
# ModelType = [True, True, True, True, True, False, False, False, False],#这里ModelType的布尔型只代表该方法是不是传统算法
# Model = [MSOA(NUMCLASSES), EEGNet(NUMCLASSES), DeepConvNet(NUMCLASSES), EEGInception(NUMCLASSES), 
#          PLNet(NUMCLASSES), HDCA(NUMCLASSES), xDAWNRG(NUMCLASSES), rLDA(NUMCLASSES), FDN(NUMCLASSES)]
ModelName =  ["EEGNet", "DeepConvNet", "EEGInception", "PLNet", "HDCA", "xDAWNRG", "rLDA","TimesNet",'EEG3d']
ModelType = [True, True, True, True, False, False, False, False,True,True],#这里ModelType的布尔型只代表该方法是不是传统算法
Model = [EEGNet(NUMCLASSES), DeepConvNet(NUMCLASSES), EEGInception(NUMCLASSES), 
         PLNet(NUMCLASSES), HDCA(NUMCLASSES), xDAWNRG(NUMCLASSES), rLDA(NUMCLASSES),TimesNet(NUMCLASSES),EEG3d(NUMCLASSES)]

if __name__ == '__main__':
    # 加载工程参数
    with open('config.json', 'r') as f:
        proj_config_para = json.load(f)

    _dataset_name = proj_config_para['DatasetName']

    sub_num = [proj_config_para['Dataset'][_dataset_name[i]]['subject_num'] for i in range(len(_dataset_name))]  #sub_num 是各数据集包含的被试数，sub_num应该是个list

    for i in range(1):
        for dataset_num in range(len(sub_num)): #len(sub_num)其实还是数据集的个数，和len(dataset_name)一样。dataset_num自然就是表示第几个数据集
        # for dataset_num in range(3,4):
            # 数据集层
            if CONTROL[dataset_num] is False: continue      #第几个数据集要不要训练，在CONTROL这个列表里以布尔数形式一一对应的规定
            # for subject_id in range(1, sub_num[dataset_num]+1): #这里+1是subject_id我们一般从S01开始。（这里缺一个如果没有这个S0X数据集跳过的代码）；左闭右开
            for subject_id in [4]:          #直接选哪些被试的数据要用
                # 数据集中的被试层
                # 完成数据加载操作
                # for idx in range(len(ModelName)):
                for idx in [8]:
                    # 对于不同模型
                    model = Model[idx].to(DEVICE)
                    print(model)
                    print(proj_config_para['DatasetName'][dataset_num], "\t", f'Subject_S{subject_id:>02d}', "\t", ModelName[idx])

                    # 训练阶段
                    TrainDataManager = DataManage(Name = ModelName[idx], DataName = _dataset_name[dataset_num], Mode = True, 
                                                SubID = subject_id, BatchSize=proj_config_para['TrainPara']['batch_size'])
                    train_data_torch, train_x_npy, train_y_npy = TrainDataManager.getData()
                    train_data_trad = dict({'x': train_x_npy, 'y': train_y_npy})
                    if ModelType[0][idx] is False:
                        # 传统方法处理
                        task_train_manage = TaskManage(subject_id, ModelName[idx], True, proj_config_para['TrainPara']['epoch'], 
                            model, train_data_trad, DEVICE, _dataset_name[dataset_num])
                        task_train_manage.goTask()
                    else:
                        # 深度学习算法处理
                        task_train_manage = TaskManage(subject_id, ModelName[idx], True, proj_config_para['TrainPara']['epoch'], 
                                                model, train_data_torch, DEVICE, _dataset_name[dataset_num])
                        task_train_manage.goTask()
                    del train_data_torch, train_x_npy, train_y_npy
                    gc.collect()
                    # subject_id = (subject_id + 1) % 14
                    # 测试阶段
                    TestDataManage = DataManage(Name = ModelName[idx], DataName = _dataset_name[dataset_num], Mode = False, 
                                                SubID = subject_id, BatchSize=proj_config_para['TestPara']['batch_size'])
                    test_data_torch, test_x_npy, test_y_npy = TestDataManage.getData()
                    test_data_trad = dict({'x': test_x_npy, 'y': test_y_npy})
                    if ModelType[0][idx] is False:
                        # 传统方法处理
                        task_train_manage = TaskManage(subject_id, ModelName[idx], False, proj_config_para['TrainPara']['epoch'], 
                            model, test_data_trad, DEVICE, _dataset_name[dataset_num])
                        task_train_manage.goTask()
                    else:
                        # 深度学习算法处理
                        task_test_manage = TaskManage(subject_id, ModelName[idx], False, proj_config_para['TestPara']['epoch'], 
                                                model, test_data_torch, DEVICE, _dataset_name[dataset_num])
                        task_test_manage.goTask()
                    
                    del test_data_torch, test_x_npy, test_y_npy
                    gc.collect()

                    evaluateManager = EvaluateManage(subject_id, _dataset_name[dataset_num], ModelName[idx])

                    score = evaluateManager.calculate_metric_score()

                    # print("AUC: ", auc)

                    # 结果以及模型保存
                    update_result(score, _dataset_name[dataset_num], ModelName[idx], subject_id, ModelType[0][idx])
                    
                
                

                
                




    