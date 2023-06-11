import math
import os
import time
import numpy as np
# from psychopy import visual, core, parallel
# from modoule1 import paradigm1
# from sklearn.preprocessing import OneHotEncoder
from preprocess import bdf_to_numpy_2
# from preprocess_copy import bdf_to_numpy,bdf_to_numpy_01   #只有tar1，没有tar2，用这个
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
import sklearn.decomposition
import sklearn.discriminant_analysis
import joblib
import pickle
from scipy import signal
from sklearn import preprocessing
import torch
import torch.nn as nn
class Mydataset():
    def __init__(self,bdf_data_path) :
        # self.paradigm_num = paradigm_num
        self.paradigm_num = int(bdf_data_path[-6:-4])
        self.bdf_data_path = bdf_data_path
        
    def filt_data(self,data, f_1=1, f_2=40, fs=256):
    # data: (N, C, T)
        wn = [f_1 * 2 / fs, f_2 * 2 / fs]
        b, a = signal.butter(3, wn, "bandpass")
        for trial in range(data.shape[0]):
            data[trial, ...] = signal.filtfilt(b, a, data[trial, ...], axis=1)


    def scale_data(self,data):
        scaler = preprocessing.StandardScaler()
        for i in range(data.shape[0]):
            data[i, :, :] = scaler.fit_transform(data[i, :, :])
    def load_data_2(self):
        #处理两个标签情况，即paradigm【】
        bdf_data_path = self.bdf_data_path
        
        rawlabel, rawdata = bdf_to_numpy_2(bdf_data_path)  #用于DPN，区分阴阳
        self.filt_data(rawdata)
        self.scale_data(rawdata)
        
        idx_0 = np.where(rawlabel==16)
        idx_1 = np.where(rawlabel==5)
        rawlabel[idx_0]=0
        rawlabel[idx_1]=1
        # np.save('./data/DPN/S06/P12/x.npy',rawdata)       #原始数据，label换成从0开始。这个x与y是给ORCNet用的
        # np.save('./data/DPN/S06/P12/y.npy',rawlabel)        
        
        # rawlabel,rawdata = bdf_to_numpy_4(bdf_data_path)      #用于直接普通处理，不区分阴阳
        # self.filt_data(rawdata)
        # self.scale_data(rawdata)
        
        # idx_0 = np.where(rawlabel==16)
        # idx_1 = np.where(rawlabel==5)
        # idx_2 = np.where(rawlabel==4)
        # rawlabel[idx_0]=0
        # rawlabel[idx_1]=1
        # rawlabel[idx_2]=2
        # np.save('./data/DPN/S06/P12/x.npy',rawdata)
        # np.save('./data/DPN/S06/P12/y.npy',rawlabel)

        train_rate = 0.75       #训练/测试比例
        x = rawdata
        y = rawlabel
        y = np.array(y, dtype=int)

        num = int(x.shape[0] * train_rate)
        x_train = x[:num]
        x_test = x[num:]
        y_train = y[:num]
        y_test = y[num:]

        #对训练集做类别平衡，测试集不做处理
        targetData_idx = np.where(y_train == 1)
        targetData = x_train[targetData_idx]
        notarData_idx = np.where(y_train == 0)
        notarData = x_train[notarData_idx]
        
        selIdx = np.array(range(notarData.shape[0]))
        np.random.shuffle(selIdx)
        targetCnt = targetData.shape[0]
        selIdx = selIdx[:targetCnt]
        notarData = notarData[selIdx]   
        data = np.concatenate((targetData, notarData), axis=0)
        
        label = np.zeros((targetCnt * 2), int)      #此种打乱方式，仅适用于0、1二分类，如果是012三分类，还需要其他方式做类别平衡或额外的处理label
        label[:targetCnt] = 1  
        
        
        shuffleIdx = np.array(range(targetCnt * 2))
        np.random.shuffle(shuffleIdx)
        label = label[shuffleIdx]
        data = data[shuffleIdx]  
        x_train = data
        y_train = label
        
        
        
        
        
        mesh0 = x_train[1,1,:]
        tempslice = np.zeros((11,256),dtype=float)
        tempslice[5,:] = mesh0
        
        EEGmesh = np.zeros((10,11,256),dtype=float)
        EEGmesh[0,4:7,:] = (x_train[1,1,:],x_train[1,32,:],x_train[1,33,:])
        EEGmesh[1,3:8,:] = (x_train[1,1,:],x_train[1,2,:],x_train[1,36,:],x_train[1,35,:],x_train[1,34,:])
        EEGmesh[2,1:10,:] = (x_train[1,6,:],x_train[1,5,:],x_train[1,4,:],x_train[1,3,:],x_train[1,37,:],x_train[1,38,:],x_train[1,39,:],x_train[1,40,:],x_train[1,41,:])
        EEGmesh[3,1:10,:] = (x_train[1,7,:],x_train[1,8,:],x_train[1,9,:],x_train[1,10,:],x_train[1,46,:],x_train[1,45,:],x_train[1,44,:],x_train[1,43,:],x_train[1,42,:])
        EEGmesh[4,1:10,:] = (x_train[1,14,:],x_train[1,13,:],x_train[1,12,:],x_train[1,11,:],x_train[1,47,:],x_train[1,48,:],x_train[1,49,:],x_train[1,50,:],x_train[1,51,:])
        EEGmesh[5,1:10,:] = (x_train[1,15,:],x_train[1,16,:],x_train[1,17,:],x_train[1,18,:],x_train[1,31,:],x_train[1,55,:],x_train[1,54,:],x_train[1,53,:],x_train[1,52,:])
        EEGmesh[6,:,:] = (x_train[1,23,:],x_train[1,22,:],x_train[1,21,:],x_train[1,20,:],x_train[1,19,:],x_train[1,30,:],x_train[1,56,:],x_train[1,57,:],x_train[1,58,:],x_train[1,59,:],x_train[1,60,:])
        EEGmesh[7,3:8,:] = (x_train[1,24,:],x_train[1,25,:],x_train[1,29,:],x_train[1,62,:],x_train[1,61,:])
        EEGmesh[8,4:7,:] = (x_train[1,26,:],x_train[1,28,:],x_train[1,63,:])
        EEGmesh[9,5,:] = (x_train[1,27,:])
        
        
        
        
        
        input = torch.tensor(EEGmesh.transpose((2,0,1)),dtype=torch.float)
        input = input.unsqueeze(0).unsqueeze(0)
        print(input.size())
        # kernel_size的第一维度的值是每次处理的图像帧数，后面是卷积核的大小
        m = nn.Conv3d(1, 8, (10, 3, 3), stride=1, padding=0)
        f = nn.Conv3d(1, 8, (11, 3, 3), padding=(5,0,0),bias=False)
        output = f(input)
        print(output.size())
        BatchNorm = nn.BatchNorm3d(8)
        output = BatchNorm(output)
        # 输出是 torch.Size([1, 3, 5, 54, 34])
        
        
        
        input = torch.tensor(EEGmesh.transpose((2,0,1)),dtype=torch.float)
        input = input.unsqueeze(0).unsqueeze(0)
        print(input.size())
        x = input
        drop_out = 0.6
    #输入数据维度（1,1,256，10，11,）
        block_1 = nn.Sequential(
            # nn.ZeroPad2d((15, 16, 0, 0)),
            nn.Conv3d(1, 8, (32, 1, 1), padding=(16,0,0),bias=False),
            # PrintLayer(),
            nn.BatchNorm3d(8),
            # nn.AvgPool3d((1,3,3))   #这里eegnet本身没有池化，可能因为二维数据通道信息太少。但这里10,11的mesh其实也只比64多了一点，不一定需要池化，不然把好不容易扩充的语义信息又失去。       
        )
        
        # block 2 and 3 are implements of Depthwise Conv and Separable Conv
        block_2 = nn.Sequential(
            nn.Conv3d(8, 16, (1,3,3), groups=8, bias=False),         #这里(8,1)中8是脑电通道，可以改成64
            nn.BatchNorm3d(16),
            nn.ELU(),           #nn.adaptive有3d。。普通的ELU()不清楚对三维数据的激活时怎么作用的？？？？
            nn.AvgPool3d((4,1,1)),       #这里eegnet本身没有stride（,stride=(1,2,2)）
            nn.Dropout(drop_out)
        )
        
        block_3 = nn.Sequential(
            # nn.ZeroPad3d((7, 8, 0, 0)),
            nn.Conv3d(16, 16, (16, 1,1), padding=(7,0,0),groups=16, bias=False),
            nn.Conv3d(16, 16, (1, 1, 1), bias=False),
            nn.BatchNorm3d(16), 
            nn.ELU(),
            nn.AvgPool3d((4, 1, 1)),        
            nn.Dropout(drop_out)
        )
        
        # fc1 = nn.Linear((5*32 * (256 // 64)), num_classes)
        fc1 = nn.Linear(17280, 2)
    

        # x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        x = block_1(x)
        # fea = x
        print('3d_outpuy',x.size())
        x = block_2(x)
        print('3d_outpuy',x.size())
        x = block_3(x)
        print('3d_outpuy',x.size())
        x = x.view(x.size(0), -1)        

        logits = fc1(x)
        # probas = nn.functional.softmax(logits, dim=1)



        drop_out = 0.6

        block_1 = nn.Sequential(
            nn.ZeroPad2d((15, 16, 0, 0)),
            nn.Conv2d(1, 8, (1, 32), bias=False),
            # PrintLayer(),
            nn.BatchNorm2d(8)
        )
        
        # block 2 and 3 are implements of Depthwise Conv and Separable Conv
        block_2 = nn.Sequential(
            nn.Conv2d(8, 16, (8, 1), groups=8, bias=False),         #这里(8,1)中8是脑电通道，可以改成64
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(drop_out)
        )
        
        block_3 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(16, 16, (1, 16), groups=16, bias=False),
            nn.Conv2d(16, 16, (1, 1), bias=False),
            nn.BatchNorm2d(16), 
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(drop_out)
        )
        
        # self.fc1 = nn.Linear((5*32 * (256 // 64)), num _classes)
        fc1 = nn.Linear(15488, 2)
        
        x= torch.tensor(x_train[:64,:,:],dtype=torch.float)
        x = x.unsqueeze(1)
        print('eeg_input',x.size())
        x = block_1(x)
        print('eeg_outpuy',x.size())

        x = block_2(x)
        print('eeg_outpuy',x.size())
        x = block_3(x)
        print('eeg_outpuy',x.size())
        x = x.view(x.size(0), -1)        

        logits = fc1(x)
        probas = nn.functional.softmax(logits, dim=1)
        
        
        
        
        
        

        exit()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        # (N, C, T) --> (N, 1, C, T)
        x_train = x_train.reshape(
            x_train.shape[0], 1, x_train.shape[1], x_train.shape[2])
        x_test = x_test.reshape(
            x_test.shape[0], 1, x_test.shape[1], x_test.shape[2])
        
        # np.save('./data/DPN/preprocess/S01/x_train.npy',x_train)
        # np.save('./data/DPN/preprocess/S01/y_train.npy',y_train)
        # np.save('./data/DPN/preprocess/S01/x_test.npy',x_test)
        # np.save('./data/DPN/preprocess/S01/y_test.npy',y_test)        
            


    
        # #测试集、训练集都做类别平衡
        # targetData_idx = np.where(rawlabel == 5)
        # targetData = rawdata[targetData_idx]
        # notarData_idx = np.where(rawlabel == 16)
        # notarData = rawdata[notarData_idx]
        # selIdx = np.array(range(notarData.shape[0]))
        # np.random.shuffle(selIdx)
        # targetCnt = targetData.shape[0]
        # selIdx = selIdx[:targetCnt]
        # notarData = notarData[selIdx] 

        # # label = np.zeros((targetCnt * 2), int)
        # # label[:targetCnt] = 1

        # shuffleIdx = np.array(range(targetCnt * 2))
        # np.random.shuffle(shuffleIdx)
        # label = label[shuffleIdx]
        # data = np.concatenate((targetData, notarData), axis=0)
        # data = data[shuffleIdx]
        

        
         
        return x_train, y_train, x_test, y_test

    def load_data_1(self):
        #处理单个标签情况，即paradigm【1】
        bdf_data_path = self.bdf_data_path
        # tar_1_label, tar_1_data,tar_2_label,tar_2_data,notar_label,notar_data = bdf_to_numpy(bdf_data_path)
        # rawlabel, rawdata = bdf_to_numpy(bdf_data_path) #P15单中正
        rawlabel,rawdata = bdf_to_numpy_01(bdf_data_path)
        
        self.filt_data(rawdata)
        self.scale_data(rawdata)
        
        idx_0 = np.where(rawlabel==16)
        idx_1 = np.where(rawlabel==4)
        rawlabel[idx_0]=0
        rawlabel[idx_1]=1
        np.save('./data/DPN/S06/P12/x.npy',rawdata)
        np.save('./data/DPN/S06/P12/y.npy',rawlabel)        
        
        ##类别平衡
        #首先把标签奇偶数分开这样
        # notar_X_data = notar_data[::2,:,:]
        # notar_Y_data = notar_data[0::2,:,:]
        notar_X_data = notar_data[:,:,:1024]
        notar_Y_data = notar_data[:,:,1024:]
        notar_train_data = np.add(notar_X_data ,notar_Y_data)/2    
        notar_data = notar_train_data[::18,:,:]
        notar_label = notar_label[::18,:] 

        tar_2_data = tar_1_data[:,:,:1024]
        tar_1_data = tar_1_data[:,:,1024:]


        target_data = np.add(tar_1_data,tar_2_data)/2

        # train_data = np.concatenate([tar_1_data,notar_data],axis=0)
        # train_label = np.concatenate([tar_1_label,notar_label],axis=0)
        train_data = np.concatenate([target_data,notar_data],axis=0)
        train_label = np.concatenate([tar_1_label,notar_label],axis=0)


        permutation = np.random.permutation(train_label.shape[0])
        shuffled_dataset = train_data[permutation, :, :]
        shuffled_labels = train_label[permutation,2]


        # train_num = len(train_label)*0.8
        train_num = int(train_label.shape[0]*0.8)
        test_num = len(train_label)*0.2

        data = shuffled_dataset[:train_num,:,:]
        label = shuffled_labels[:train_num]

        data = data[::2,:,:]
        label = label[::2]

        test_data = shuffled_dataset[train_num:,:,:]
        test_label = shuffled_labels[train_num:]

        test_label = test_label[::2]
        test_data = test_data[::2,:,:]
        return data, label, test_data, test_label
    
    
    def load_data_3(self):
        #处理两个标签情况，即paradigm【】
        bdf_data_path = self.bdf_data_path
        tar_1_label, tar_1_data,tar_2_label,tar_2_data,notar_label,notar_data = bdf_to_numpy_3(bdf_data_path)
        
        
        #切分训练集和测试集，训练集做类别平衡，测试不可以做空间拼接之外的任何操作
        #一共6*8=48个trail，阴阳各有6*4*5=120个目标，有6*4*45=1080个非目标，其中非目标没有在标签上区分阴阳
        train_num = (tar_1_label.shape[0]*2 + notar_label.shape[0])*0.8
        
        
        target_data = np.concatenate((tar_1_data,tar_2_data),axis=0)#标签4、5都当做目标，不做数据叠加

        ##类别平衡
        #首先把标签奇偶数分开这样

        # notar_train_data = np.concatenate((notar_X_data,notar_Y_data),axis=1)
        notar_data = notar_data[::9,:,:]
        notar_label = notar_label[::9,:]


        train_data = np.concatenate([target_data,notar_data],axis=0)
        train_label = np.concatenate([tar_1_label,tar_2_label,notar_label],axis=0)

        
        permutation = np.random.permutation(train_label.shape[0])
        shuffled_dataset = train_data[permutation, :, :]
        shuffled_labels = train_label[permutation,2]            #第2维是trigger数值，因为bdf读出来第1维都是0


        # train_num = len(train_label)*0.8
        train_num = int(train_label.shape[0]*0.8)
        test_num = len(train_label)*0.2

        data = shuffled_dataset[:train_num,:,:]
        label = shuffled_labels[:train_num]

        test_data = shuffled_dataset[train_num:,:,:]
        test_label = shuffled_labels[train_num:]         
        
         
        return data, label, test_data, test_label
    
    def load_data(self):
        if self.paradigm_num in [1,5,7,8,15]:
            data, label, test_data, test_label = self.load_data_1()
        elif self.paradigm_num in [6,9,10,12,13]:
            data, label, test_data, test_label = self.load_data_2()
        elif self.paradigm_num in [14]:
            data, label, test_data, test_label = self.load_data_3()
        else:
            print('范式类型不存在')
        return data, label, test_data, test_label 

class toEEGmesh():
    def __init__(self,data) :
        # self.paradigm_num = paradigm_num
        self.data = data
    def to3D(self):
        self.data




if __name__ == '__main__':
    bdf_data_path = 'D:/2023/Dual&negative-RSVP/实验信息/S06/0012.bdf'
    makedata = Mydataset(bdf_data_path)
    
    data, label, test_data, test_label = makedata.load_data()
    exit()