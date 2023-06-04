'''
Author: Tammie li
Description: Define model
FilePath: \model.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from ns_layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
# from ns_layers.SelfAttention_Family import DSAttention, AttentionLayer
# from layers.Embed import DataEmbedding
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if(self.max_norm != None):
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)


class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if(self.max_norm != None):
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)



class MSOA(nn.Module):
    '''
    @description: multi-task self-supervised adaptation algorithm
    @inputparams: 
    n_class_main: Number of classes of main task
    n_class_aux_tem: Number of classes of auxiliary task VTO
    n_class_aux_spa: Number of classes of auxiliary task MSP
    channels:
    @Hyperparams: 
    n_kernel_t: Number of convolution kernels in temporal feature extraction block
    n_kernel_s: Number of convolution kernels in spatial feature extraction block
    dropout: default=0.5
    kernel_length: Length of  temporal convolution kernel, default=64
    '''
    def __init__(self, n_class_primary, T = 256, channels=64, n_kernel_t=8, n_kernel_s=16, dropout=0.5, kernel_length=32):
        super(MSOA, self).__init__()
        self.n_class_primary = n_class_primary
        self.channels = channels
        self.n_kernel_t = n_kernel_t
        self.n_kernel_s = n_kernel_s
        self.dropout = dropout
        self.kernel_length = kernel_length
   
        # Feature extractor (EEGNet Architecture)
        # self.temp_regular_conv = nn.Sequential(
        #     nn.ZeroPad2d((self.kernel_length//2-1, self.kernel_length//2, 0, 0)),
        #     nn.Conv2d(1, self.n_kernel_t, (1, self.kernel_length), bias=False),
        #     nn.BatchNorm2d(self.n_kernel_t)
        # )

        # self.spatial_depth_conv = nn.Sequential(
        #     Conv2dWithConstraint(self.n_kernel_t, self.n_kernel_s, (self.channels, 1), groups=self.n_kernel_t, bias=False),
        #     nn.BatchNorm2d(self.n_kernel_s),
        #     nn.ELU(), 
        #     nn.AvgPool2d((1, 4)),
        #     nn.Dropout(self.dropout)
        # )

        # self.fusion_point_conv = nn.Sequential(
        #     nn.ZeroPad2d((self.kernel_length//8-1, self.kernel_length//8, 0, 0)),
        #     nn.Conv2d(self.n_kernel_s, self.n_kernel_s, (1, self.kernel_length//4), groups=self.n_kernel_s, bias=False),
        #     nn.Conv2d(self.n_kernel_s, self.n_kernel_s, (1, 1), bias=False),
        #     nn.BatchNorm2d(self.n_kernel_s),
        #     nn.ELU(),
        #     nn.AvgPool2d((1, 8)),
        #     nn.Dropout(self.dropout)
        # )

        self.block0 = nn.Sequential()
        self.block0.add_module('conv1', nn.Conv2d(1, 25, (1, 5), bias=False))

        self.block1 = nn.Sequential()
        self.block1.add_module('conv2', nn.Conv2d(25, 25, (64, 1), bias=False))
        self.block1.add_module('norm1', nn.BatchNorm2d(25))
        self.block1.add_module('act1', nn.ELU())

        self.block1.add_module('pool1', nn.MaxPool2d((1, 2), stride=(1, 2)))
        self.block1.add_module('drop1', nn.Dropout(p=0.5))

        self.block2 = nn.Sequential()

        self.block2.add_module('conv3', nn.Conv2d(25, 50, (1, 5), bias=False))
        self.block2.add_module('norm2', nn.BatchNorm2d(50))
        self.block2.add_module('act2', nn.ELU())

        self.block2.add_module('pool2', nn.MaxPool2d((1, 2), stride=(1, 2)))
        self.block2.add_module('drop2', nn.Dropout(p=0.5))
        
        self.block3 = nn.Sequential()

        self.block3.add_module('conv4', nn.Conv2d(50, 100, (1, 5), bias=False))
        self.block3.add_module('norm3', nn.BatchNorm2d(100))
        self.block3.add_module('act3', nn.ELU())

        self.block3.add_module('pool3', nn.MaxPool2d((1, 2), stride=(1, 2)))
        self.block3.add_module('drop3', nn.Dropout(p=0.5))

        self.block4 = nn.Sequential()

        self.block4.add_module('conv5', nn.Conv2d(100, 200, (1, 5), bias=False))
        self.block4.add_module('norm4', nn.BatchNorm2d(200))
        self.block4.add_module('act4', nn.ELU())

        self.block4.add_module('pool4', nn.MaxPool2d((1, 2), stride=(1, 2)))
        self.block4.add_module('drop4', nn.Dropout(p=0.5))

        # Fully-connected layer
        # self.primary_task_classification = LinearWithConstraint(self.n_kernel_s*T//32, n_class_primary)
        # self.vto_task_projection_head = LinearWithConstraint(self.n_kernel_s*T//32, 16)
        # self.vto_task_classification = LinearWithConstraint(16, 9)

        # self.msp_task_projection_head = LinearWithConstraint(self.n_kernel_s*T//32, 16)
        # self.msp_task_classification = LinearWithConstraint(16, 8)

        self.primary_task_classification = nn.Linear(200*12, n_class_primary)
        self.vto_task_projection_head = nn.Linear(200*12, 16)
        self.vto_task_classification = nn.Linear(16, 9)

        self.msp_task_projection_head = nn.Linear(200*12, 16)
        self.msp_task_classification = nn.Linear(16, 8)


    def forward(self, x, stage_name, task_name):
        '''
        @description: Complete the corresponding task according to the task tag
        '''
        # extract features
        x = torch.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2]))
        
        # x = self.temp_regular_conv(x)
        # x = self.spatial_depth_conv(x)

        # x = self.fusion_point_conv(x)
        if stage_name == "testStageI":
            with torch.no_grad():
                x = self.block0(x)
        else:
            x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        fea = x.view(x.size(0), -1)

        if stage_name == "trainStage":
            if task_name == "main":
                logits = self.primary_task_classification(fea)
            elif task_name == "vto":
                logits = self.vto_task_projection_head(fea)
                logits = self.vto_task_classification(logits)
            elif task_name == "msp":
                logits = self.msp_task_projection_head(fea)
                logits = self.msp_task_classification(logits)
            else:
                assert("TaskName Error!")

            pred = F.softmax(logits, dim = 1)

            return pred

        elif stage_name == "testStageI":
            if task_name == "vto":
                with torch.no_grad():
                    logits = self.vto_task_projection_head(fea)
                    logits = self.vto_task_classification(logits)
            elif task_name == "msp":
                with torch.no_grad():
                    logits = self.msp_task_projection_head(fea)
                    logits = self.msp_task_classification(logits)
            else:
                assert("TaskName Error!")
            
            pred = F.softmax(logits, dim = 1)
            return pred

        elif stage_name == "testStageII":
            with torch.no_grad():
                primary = self.primary_task_classification(fea)
                pred = F.softmax(primary, dim = 1)
            return pred
        elif stage_name == "GetFeature":
            return fea        
        elif stage_name == "GetHeadVTO":
            head_vto = self.vto_task_projection_head(fea)
            return head_vto
        elif stage_name == "GetHeadMSP":
            head_msp = self.msp_task_projection_head(fea)
            return head_msp
        else:
            assert("Please enter the correct stage name!")


class DeepConvNet(nn.Module):
    def __init__(self, num_classes):
        super(DeepConvNet, self).__init__()
        self.block1 = nn.Sequential()
        self.block1.add_module('conv1', nn.Conv2d(1, 25, (1, 5), bias=False))

        self.block1.add_module('conv2', nn.Conv2d(25, 25, (64, 1), bias=False))
        self.block1.add_module('norm1', nn.BatchNorm2d(25))
        self.block1.add_module('act1', nn.ELU())

        self.block1.add_module('pool1', nn.MaxPool2d((1, 2), stride=(1, 2)))
        self.block1.add_module('drop1', nn.Dropout(p=0.5))

        self.block2 = nn.Sequential()

        self.block2.add_module('conv3', nn.Conv2d(25, 50, (1, 5), bias=False))
        self.block2.add_module('norm2', nn.BatchNorm2d(50))
        self.block2.add_module('act2', nn.ELU())

        self.block2.add_module('pool2', nn.MaxPool2d((1, 2), stride=(1, 2)))
        self.block2.add_module('drop2', nn.Dropout(p=0.5))
        
        self.block3 = nn.Sequential()

        self.block3.add_module('conv4', nn.Conv2d(50, 100, (1, 5), bias=False))
        self.block3.add_module('norm3', nn.BatchNorm2d(100))
        self.block3.add_module('act3', nn.ELU())

        self.block3.add_module('pool3', nn.MaxPool2d((1, 2), stride=(1, 2)))
        self.block3.add_module('drop3', nn.Dropout(p=0.5))

        self.block4 = nn.Sequential()

        self.block4.add_module('conv5', nn.Conv2d(100, 200, (1, 5), bias=False))
        self.block4.add_module('norm4', nn.BatchNorm2d(200))
        self.block4.add_module('act4', nn.ELU())

        self.block4.add_module('pool4', nn.MaxPool2d((1, 2), stride=(1, 2)))
        self.block4.add_module('drop4', nn.Dropout(p=0.5))

        self.classify = nn.Sequential(
            nn.Linear(200*12, 2)
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        res = x.view(x.size(0), -1)

        out = self.classify(res)
        return out

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)      #print(x.shape)
        return x
class EEGNet(nn.Module):
    def __init__(self, num_classes):
        super(EEGNet, self).__init__()
        self.drop_out = 0.6

        self.block_1 = nn.Sequential(
            nn.ZeroPad2d((15, 16, 0, 0)),
            nn.Conv2d(1, 8, (1, 32), bias=False),
            # PrintLayer(),
            nn.BatchNorm2d(8)
        )
        
        # block 2 and 3 are implements of Depthwise Conv and Separable Conv
        self.block_2 = nn.Sequential(
            nn.Conv2d(8, 16, (8, 1), groups=8, bias=False),         #这里(8,1)中8是脑电通道，可以改成64
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(self.drop_out)
        )
        
        self.block_3 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(16, 16, (1, 16), groups=16, bias=False),
            nn.Conv2d(16, 16, (1, 1), bias=False),
            nn.BatchNorm2d(16), 
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.drop_out)
        )
        
        # self.fc1 = nn.Linear((5*32 * (256 // 64)), num_classes)
        self.fc1 = nn.Linear(15488, num_classes)
    
    def forward(self, x):
        # x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        x = self.block_1(x)
        # fea = x

        x = self.block_2(x)
        x = self.block_3(x)

        x = x.view(x.size(0), -1)        

        logits = self.fc1(x)
        probas = F.softmax(logits, dim=1)
        return probas


class EEGInception(nn.Module):
    def __init__(self, num_classes, C=64, T=256, drop_out=0.5):
        super(EEGInception, self).__init__()
        self.T = T
        self.C = C
        self.drop_out = 0.5
        # input size: (N, 1, C, T)
        self.time_block_11 = nn.Sequential(
            nn.ZeroPad2d((31, 32, 0, 0)),
            nn.Conv2d(1, 8, (1, 64)),   #时序分析
            nn.BatchNorm2d(8),
            nn.Dropout(self.drop_out), 
            nn.Conv2d(8, 16, (self.C, 1), groups=8),    #空间分析，片卷积（depthwse）
            nn.BatchNorm2d(16),
            nn.Dropout(self.drop_out)
        )
        self.time_block_12 = nn.Sequential(
            nn.ZeroPad2d((15, 16, 0, 0)),
            nn.Conv2d(1, 8, (1, 32)),
            nn.BatchNorm2d(8),
            nn.Dropout(self.drop_out), 
            nn.Conv2d(8, 16, (self.C, 1), groups=8),
            nn.BatchNorm2d(16),
            nn.Dropout(self.drop_out)
        )
        self.time_block_13 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(1, 8, (1, 16)),
            nn.BatchNorm2d(8),
            nn.Dropout(self.drop_out), 
            nn.Conv2d(8, 16, (self.C, 1), groups=8),
            nn.BatchNorm2d(16),
            nn.Dropout(self.drop_out)
        )
        
        self.time_block_21 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(48, 8, (1, 16)),
            nn.BatchNorm2d(8),
            nn.Dropout(self.drop_out)
        )
        self.time_block_22 = nn.Sequential(
            nn.ZeroPad2d((3, 4, 0, 0)),
            nn.Conv2d(48, 8, (1, 8)),
            nn.BatchNorm2d(8),
            nn.Dropout(self.drop_out)
        )
        self.time_block_23 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 0, 0)),
            nn.Conv2d(48, 8, (1, 4)),
            nn.BatchNorm2d(8),
            nn.Dropout(self.drop_out)
        )
        
        self.time_block_3 = nn.Sequential(
            nn.ZeroPad2d((3, 4, 0, 0)),
            nn.Conv2d(24, 12, (1, 8)),
            nn.BatchNorm2d(12),
            nn.Dropout(self.drop_out)
        )
        
        self.time_block_4 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 0, 0)),
            nn.Conv2d(12, 6, (1, 4)),
            nn.BatchNorm2d(6),
            nn.Dropout(self.drop_out)
        )
        
        self.pool_1 = nn.AvgPool2d((1, 4))
        self.pool_2 = nn.AvgPool2d((1, 2))
        
        self.fc = nn.Linear(self.T // (4 * 2 * 2 * 2) * 6, num_classes)
    
    def forward(self, x):
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        x_11 = self.time_block_11(x)
        x_12 = self.time_block_12(x)
        x_13 = self.time_block_13(x)
        x = torch.cat((x_11, x_12, x_13), dim=1)
        x = self.pool_1(x)
        
        x_21 = self.time_block_21(x)
        x_22 = self.time_block_22(x)
        x_23 = self.time_block_23(x)
        x = torch.cat((x_21, x_22, x_23), dim=1)
        x = self.pool_2(x)
        
        x = self.time_block_3(x)
        x = self.pool_2(x)
        x = self.time_block_4(x)
        x = self.pool_2(x)
        
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return probas


class PLNet(nn.Module):
    def __init__(self, num_classes):
        super(PLNet, self).__init__()
        self.block1 = torch.nn.Sequential(
            Conv2dWithConstraint(
                in_channels=1,
                out_channels=8,
                kernel_size=(1, 32),
                bias=False,
                stride=(1, 4),
                max_norm=0.5
            ),   # F1, C, T
            torch.nn.BatchNorm2d(
                num_features=8
            ),
            torch.nn.ELU()
        )

        tmp = torch.Tensor(np.ones((1, 1, 64, 256), dtype=float))
        tmp = self.block1(tmp)
        # Permute
        tmp = tmp.view(1, tmp.shape[3], tmp.shape[2], tmp.shape[1])

        self.block2 = torch.nn.Sequential(
            # DepthwiseConv2D
            Conv2dWithConstraint(
                in_channels=tmp.shape[1],
                out_channels=tmp.shape[1],
                kernel_size=(64, 1),
                max_norm=0.5,
                stride=1,
                groups=tmp.shape[1],
                bias=False
            )
        )
        tmp = self.block2(tmp)

        # Permute
        tmp = tmp.view(1, tmp.shape[3], tmp.shape[2], tmp.shape[1])
        self.block3 = nn.Sequential(
            nn.BatchNorm2d(
                num_features=8),
            nn.ELU(),
            nn.Dropout(0.5),
            # SeparableConv2D
            Conv2dWithConstraint(
                in_channels=tmp.shape[1],
                out_channels=tmp.shape[1] * 2,
                kernel_size=(1, 9),
                groups=tmp.shape[1],
                bias=False,
                stride=1,
                max_norm=0.5
            ),
            Conv2dWithConstraint(
                in_channels=tmp.shape[1] * 2,
                out_channels=tmp.shape[1] * 2,
                kernel_size=(1, 1),
                bias=False,
                stride=1,
                max_norm=0.5
            ),
            torch.nn.BatchNorm2d(
                num_features=tmp.shape[1] * 2),
            torch.nn.ELU(),
        )
        tmp = self.block3(tmp)
        self.pooling = torch.nn.AdaptiveAvgPool2d(1)
        tmp = self.pooling(tmp)
        self.drop_out = torch.nn.Dropout(0.5)

        self.classifier = torch.nn.Sequential(
            LinearWithConstraint(tmp.shape[1], num_classes, max_norm=0.1)
        )

    def forward(self, data):
        batch_size = len(data)
        data = data.reshape(data.shape[0], 1, data.shape[1], data.shape[2])
        data = self.block1(data)
        data = data.view(
            batch_size, data.shape[3], data.shape[2], data.shape[1])
        data = self.block2(data)
        data = data.view(
            batch_size, data.shape[3], data.shape[2], data.shape[1])
        data = self.block3(data)
        data = self.pooling(data)
        data = self.drop_out(data)
        data = data.view(batch_size, -1)
        return self.classifier(data)

class HDCA(nn.Module):
    def __init__(self, num_classes):
        super(HDCA, self).__init__()
        pass
    
    def forward(self, x):
        pass

class rLDA(nn.Module):
    def __init__(self, num_classes):
        super(rLDA, self).__init__()
        pass
    
    def forward(self, x):
        pass

class xDAWNRG(nn.Module):
    def __init__(self, num_classes):
        super(xDAWNRG, self).__init__()
        pass
    
    def forward(self, x):
        pass





class Projector(nn.Module):
    '''
    MLP to learn the De-stationary factors
    '''
    def __init__(self, enc_in, seq_len, hidden_dims, hidden_layers, output_dim, kernel_size=3):
        super(Projector, self).__init__()

        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.series_conv = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=kernel_size, padding=padding, padding_mode='circular', bias=False)

        layers = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers-1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i+1]), nn.ReLU()]
        
        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, stats):
        # x:     B x S x E
        # stats: B x 1 x E
        # y:     B x O
        batch_size = x.shape[0]
        x = self.series_conv(x)          # B x 1 x E
        x = torch.cat([x, stats], dim=1) # B x 2 x E
        x = x.view(batch_size, -1) # B x 2E
        y = self.backbone(x)       # B x O

        return y

class FDN(nn.Module):
    '''
    @description: Feature Decomposition for Reducing Negative Transfer: A Novel Multi-task Learning Method for Recommender System
    @inputparams: EEGNet

    @Hyperparams: 

    '''
    def __init__(self, n_class_primary, T = 256, channels=64, n_kernel_t=8, n_kernel_s=16, dropout=0.5, kernel_length=32):
        super(FDN, self).__init__()

        self.n_class_primary = n_class_primary
        self.channels = channels
        self.n_kernel_t = n_kernel_t
        self.n_kernel_s = n_kernel_s
        self.dropout = dropout
        self.kernel_length = kernel_length

        
        self.block_shared_feature_extractor = nn.Sequential(
            # 原block1
            nn.ZeroPad2d((15, 16, 0, 0)),
            nn.Conv2d(1, 8, (1, 32), bias=False),
            nn.BatchNorm2d(8),
            # 原block2
            nn.Conv2d(8, 16, (64, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(self.dropout)
        )
        self.block_specific_main_feature_extractor = nn.Sequential(
            # 原block1
            nn.ZeroPad2d((15, 16, 0, 0)),
            nn.Conv2d(1, 8, (1, 32), bias=False),
            nn.BatchNorm2d(8),
            # 原block2
            nn.Conv2d(8, 16, (64, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(self.dropout)
        )
        self.block_specific_mtr_feature_extractor = nn.Sequential(
            # 原block1
            nn.ZeroPad2d((15, 16, 0, 0)),
            nn.Conv2d(1, 8, (1, 32), bias=False),
            nn.BatchNorm2d(8),
            # 原block2
            nn.Conv2d(8, 16, (64, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(self.dropout)
        )
        self.block_specific_msr_feature_extractor = nn.Sequential(
            # 原block1
            nn.ZeroPad2d((15, 16, 0, 0)),
            nn.Conv2d(1, 8, (1, 32), bias=False),
            nn.BatchNorm2d(8),
            # 原block2
            nn.Conv2d(8, 16, (64, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(self.dropout)
        )

        self.block_feature_fusion = nn.Sequential(
            nn.ZeroPad2d((self.kernel_length//8-1, self.kernel_length//8, 0, 0)),
            nn.Conv2d(self.n_kernel_s, self.n_kernel_s, (1, self.kernel_length//4), groups=self.n_kernel_s, bias=False),
            nn.Conv2d(self.n_kernel_s, self.n_kernel_s, (1, 1), bias=False),
            nn.BatchNorm2d(self.n_kernel_s),
            nn.ELU()
            # nn.AvgPool2d((1, 8)),
            # nn.Dropout(self.dropout)
        )
        self.main_task_projection_head =  nn.Sequential(
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.dropout)
        )
        self.vto_task_projection_head =  nn.Sequential(
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.dropout)
        )
        self.msp_task_projection_head =  nn.Sequential(
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.dropout)
        )

        # Fully-connected layer
        self.primary_task_classifier = nn.Sequential(
            nn.Linear(self.n_kernel_s*T//32, self.n_class_primary)
        )
        self.vto_task_classifier = nn.Sequential(
            nn.Linear(self.n_kernel_s*T//32, 9)
        )
        self.msp_task_classifier = nn.Sequential(
            nn.Linear(self.n_kernel_s*T//32, 8)
        )

    def calculate_orthogonal_constraint(self, feature_1, feature_2):
        assert feature_1.shape == feature_2.shape, "the dimension of two matrix is not equal"
        N, C, H, W = feature_1.shape
        feature_1, feature_2 = torch.reshape(feature_1, (N*C, H, W)), torch.reshape(feature_2, (N*C, H, W))
        weight_squared = torch.bmm(feature_1, feature_2.permute(0, 2, 1))
        # weight_squared = torch.norm(weight_squared, p=2)
        ones = torch.ones(N*C, H, H, dtype=torch.float32).to(torch.device('cuda:0'))
        diag = torch.eye(H, dtype=torch.float32).to(torch.device('cuda:0'))

        loss = ((weight_squared * (ones - diag)) ** 2).sum()
        return loss

    def forward(self, x, task_name):
        '''
        @description: Complete the corresponding task according to the task tag
        '''
        # extract features
        x = torch.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2]))
        
        fea_shared_extract = self.block_shared_feature_extractor(x)
        fea_after_fusion = self.block_feature_fusion(fea_shared_extract)

        if task_name == "main":
            # 推理过程
            fea_specific_main = self.block_specific_main_feature_extractor(x)
            fea_main = fea_specific_main + fea_after_fusion
            fea_main = self.main_task_projection_head(fea_main)
            fea_main = fea_main.view(fea_main.size(0), -1)
            logits_main = self.primary_task_classifier(fea_main)
            pred_main = F.softmax(logits_main, dim = 1)
            # 损失计算
            orthogonal_constraint = self.calculate_orthogonal_constraint(fea_specific_main, fea_shared_extract)

            return pred_main, orthogonal_constraint

        elif task_name == "vto":
            # 推理过程
            fea_specific_vto = self.block_specific_mtr_feature_extractor(x)
            fea_vto = (fea_specific_vto + fea_after_fusion)
            fea_vto = self.vto_task_projection_head(fea_vto)
            fea_vto = fea_vto.view(fea_vto.size(0), -1)
            logits_vto = self.vto_task_classifier(fea_vto)
            pred_vto = F.softmax(logits_vto, dim = 1)
            # 损失计算
            orthogonal_constraint = self.calculate_orthogonal_constraint(fea_specific_vto, fea_shared_extract)

            return pred_vto, orthogonal_constraint

        elif task_name == "msp":
            # 推理过程
            fea_specific_msp = self.block_specific_msr_feature_extractor(x)
            fea_msp = (fea_specific_msp + fea_after_fusion)
            fea_msp = self.msp_task_projection_head(fea_msp)
            fea_msp = fea_msp.view(fea_msp.size(0), -1)
            logits_msp = self.msp_task_classifier(fea_msp)
            pred_msp = F.softmax(logits_msp, dim = 1)
            # 损失计算
            orthogonal_constraint = self.calculate_orthogonal_constraint(fea_specific_msp, fea_shared_extract)

            return pred_msp, orthogonal_constraint
        else:
            assert("TaskName Error!")

class DPNet(nn.Module):
    def __init__(self, num_classes, C=64, T=256, drop_out=0.5):
        super(EEGInception, self).__init__()
        self.T = T
        self.C = C
        self.drop_out = 0.5
        # input size: (N, 1, C, T)
        self.time_block_11 = nn.Sequential(
            nn.ZeroPad2d((31, 32, 0, 0)),
            nn.Conv2d(1, 8, (1, 64)),   #时序分析
            nn.BatchNorm2d(8),
            nn.Dropout(self.drop_out), 
            nn.Conv2d(8, 16, (self.C, 1), groups=8),    #空间分析，片卷积（depthwse）
            nn.BatchNorm2d(16),
            nn.Dropout(self.drop_out)
        )
        self.time_block_12 = nn.Sequential(
            nn.ZeroPad2d((15, 16, 0, 0)),
            nn.Conv2d(1, 8, (1, 32)),
            nn.BatchNorm2d(8),
            nn.Dropout(self.drop_out), 
            nn.Conv2d(8, 16, (self.C, 1), groups=8),
            nn.BatchNorm2d(16),
            nn.Dropout(self.drop_out)
        )
        self.time_block_13 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(1, 8, (1, 16)),
            nn.BatchNorm2d(8),
            nn.Dropout(self.drop_out), 
            nn.Conv2d(8, 16, (self.C, 1), groups=8),
            nn.BatchNorm2d(16),
            nn.Dropout(self.drop_out)
        )
        
        self.time_block_21 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(48, 8, (1, 16)),
            nn.BatchNorm2d(8),
            nn.Dropout(self.drop_out)
        )
        self.time_block_22 = nn.Sequential(
            nn.ZeroPad2d((3, 4, 0, 0)),
            nn.Conv2d(48, 8, (1, 8)),
            nn.BatchNorm2d(8),
            nn.Dropout(self.drop_out)
        )
        self.time_block_23 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 0, 0)),
            nn.Conv2d(48, 8, (1, 4)),
            nn.BatchNorm2d(8),
            nn.Dropout(self.drop_out)
        )
        
        self.time_block_3 = nn.Sequential(
            nn.ZeroPad2d((3, 4, 0, 0)),
            nn.Conv2d(24, 12, (1, 8)),
            nn.BatchNorm2d(12),
            nn.Dropout(self.drop_out)
        )
        
        self.time_block_4 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 0, 0)),
            nn.Conv2d(12, 6, (1, 4)),
            nn.BatchNorm2d(6),
            nn.Dropout(self.drop_out)
        )
        
        self.pool_1 = nn.AvgPool2d((1, 4))
        self.pool_2 = nn.AvgPool2d((1, 2))
        
        self.fc = nn.Linear(self.T // (4 * 2 * 2 * 2) * 6, num_classes)
    
# class FDN(nn.Module):
#     '''
#     @description: Feature Decomposition for Reducing Negative Transfer: A Novel Multi-task Learning Method for Recommender System
#     @inputparams:PLNet

#     @Hyperparams: 

#     '''
#     def __init__(self, n_class_primary, T = 256, channels=64, n_kernel_t=8, n_kernel_s=16, dropout=0.33, kernel_length=32):
#         super(FDN, self).__init__()

#         self.n_class_primary = n_class_primary
#         self.channels = channels
#         self.n_kernel_t = n_kernel_t
#         self.n_kernel_s = n_kernel_s
#         self.dropout = dropout
#         self.kernel_length = kernel_length

        
#         self.block_shared_feature_extractor_1 = nn.Sequential(
#             Conv2dWithConstraint(in_channels=1, out_channels=8, kernel_size=(1, 32), bias=False, stride=(1, 4), max_norm=0.5), 
#             torch.nn.BatchNorm2d(num_features=8),
#             torch.nn.ELU(),
#         )
              
#         tmp_shared_1 = torch.Tensor(np.ones((1, 1, 64, 256), dtype=float))
#         tmp_shared_1 = self.block_shared_feature_extractor_1(tmp_shared_1)
#         # Permute
#         tmp_shared_1 = tmp_shared_1.view(1, tmp_shared_1.shape[3], tmp_shared_1.shape[2], tmp_shared_1.shape[1])
        
#         self.block_shared_feature_extractor_2 = torch.nn.Sequential(
#             Conv2dWithConstraint(in_channels=tmp_shared_1.shape[1], out_channels=tmp_shared_1.shape[1], 
#                                  kernel_size=(64, 1), max_norm=0.5, stride=1, groups=tmp_shared_1.shape[1], bias=False)
#         )
#         tmp_shared_2 = self.block_shared_feature_extractor_2(tmp_shared_1)
#         tmp_shared_2 = tmp_shared_2.view(1, tmp_shared_2.shape[3], tmp_shared_2.shape[2], tmp_shared_2.shape[1])

#         # main
#         self.block_specific_main_feature_extractor_1 = nn.Sequential(
#             Conv2dWithConstraint(in_channels=1, out_channels=8, kernel_size=(1, 32), bias=False, stride=(1, 4), max_norm=0.5), 
#             torch.nn.BatchNorm2d(num_features=8),
#             torch.nn.ELU(),
#         )
              
#         tmp_specific_main_1 = torch.Tensor(np.ones((1, 1, 64, 256), dtype=float))
#         tmp_specific_main_1 = self.block_specific_main_feature_extractor_1(tmp_specific_main_1)
#         # Permute
#         tmp_specific_main_1 = tmp_specific_main_1.view(1, tmp_specific_main_1.shape[3], tmp_specific_main_1.shape[2], tmp_specific_main_1.shape[1])
        
#         self.block_specific_main_feature_extractor_2 = torch.nn.Sequential(
#             Conv2dWithConstraint(in_channels=tmp_specific_main_1.shape[1], out_channels=tmp_specific_main_1.shape[1], 
#                                  kernel_size=(64, 1), max_norm=0.5, stride=1, groups=tmp_specific_main_1.shape[1], bias=False)
#         )
#         tmp_specific_main_2 = self.block_specific_main_feature_extractor_2(tmp_specific_main_1)
#         tmp_specific_main_2 = tmp_specific_main_2.view(1, tmp_specific_main_2.shape[3], tmp_specific_main_2.shape[2], tmp_specific_main_2.shape[1])

#         # vto
#         self.block_specific_vto_feature_extractor_1 = nn.Sequential(
#             Conv2dWithConstraint(in_channels=1, out_channels=8, kernel_size=(1, 32), bias=False, stride=(1, 4), max_norm=0.5), 
#             torch.nn.BatchNorm2d(num_features=8),
#             torch.nn.ELU(),
#         )
              
#         tmp_specific_vto_1 = torch.Tensor(np.ones((1, 1, 64, 256), dtype=float))
#         tmp_specific_vto_1 = self.block_specific_vto_feature_extractor_1(tmp_specific_vto_1)
#         # Permute
#         tmp_specific_vto_1 = tmp_specific_vto_1.view(1, tmp_specific_vto_1.shape[3], tmp_specific_vto_1.shape[2], tmp_specific_vto_1.shape[1])
        
#         self.block_specific_vto_feature_extractor_2 = torch.nn.Sequential(
#             Conv2dWithConstraint(in_channels=tmp_specific_vto_1.shape[1], out_channels=tmp_specific_vto_1.shape[1], 
#                                  kernel_size=(64, 1), max_norm=0.5, stride=1, groups=tmp_specific_vto_1.shape[1], bias=False)
#         )
#         tmp_specific_vto_2 = self.block_specific_vto_feature_extractor_2(tmp_specific_vto_1)
#         tmp_specific_vto_2 = tmp_specific_vto_2.view(1, tmp_specific_vto_2.shape[3], tmp_specific_vto_2.shape[2], tmp_specific_vto_2.shape[1])

#         # msp
#         self.block_specific_msp_feature_extractor_1 = nn.Sequential(
#             Conv2dWithConstraint(in_channels=1, out_channels=8, kernel_size=(1, 32), bias=False, stride=(1, 4), max_norm=0.5), 
#             torch.nn.BatchNorm2d(num_features=8),
#             torch.nn.ELU(),
#         )
              
#         tmp_specific_msp_1 = torch.Tensor(np.ones((1, 1, 64, 256), dtype=float))
#         tmp_specific_msp_1 = self.block_specific_msp_feature_extractor_1(tmp_specific_msp_1)
#         # Permute
#         tmp_specific_msp_1 = tmp_specific_msp_1.view(1, tmp_specific_msp_1.shape[3], tmp_specific_msp_1.shape[2], tmp_specific_msp_1.shape[1])
        
#         self.block_specific_msp_feature_extractor_2 = torch.nn.Sequential(
#             Conv2dWithConstraint(in_channels=tmp_specific_msp_1.shape[1], out_channels=tmp_specific_msp_1.shape[1], 
#                                  kernel_size=(64, 1), max_norm=0.5, stride=1, groups=tmp_specific_msp_1.shape[1], bias=False)
#         )
#         tmp_specific_msp_2 = self.block_specific_msp_feature_extractor_2(tmp_specific_msp_1)
#         tmp_specific_msp_2 = tmp_specific_msp_2.view(1, tmp_specific_msp_2.shape[3], tmp_specific_msp_2.shape[2], tmp_specific_msp_2.shape[1])
        
#         tmp_specific_msp_2 = torch.cat([tmp_specific_msp_2, tmp_specific_msp_2], dim=1)
#         self.block_feature_fusion = nn.Sequential(
#             nn.BatchNorm2d(num_features=16),
#             nn.ELU(),
#             nn.Dropout(0.5),
#             # SeparableConv2D
#             Conv2dWithConstraint(in_channels=tmp_specific_msp_2.shape[1], out_channels=tmp_specific_msp_2.shape[1] * 2, 
#                                  kernel_size=(1, 9), groups=tmp_specific_msp_2.shape[1], bias=False, stride=1, max_norm=0.5),
#             Conv2dWithConstraint(in_channels=tmp_specific_msp_2.shape[1] * 2, out_channels=tmp_specific_msp_2.shape[1] * 2, 
#                                  kernel_size=(1, 1), bias=False, stride=1, max_norm=0.),
#             torch.nn.BatchNorm2d(num_features=tmp_specific_msp_2.shape[1] * 2),
#             torch.nn.ELU(),
#         )
#         fusion_feature = self.block_feature_fusion(tmp_specific_msp_2)

#         # fea_after_fusion = self.pooling(fea_after_fusion)
#         # fea_after_fusion = self.drop_out(fea_after_fusion)

#         self.main_task_projection_head =  nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Dropout(0.5)
#         )
#         self.vto_task_projection_head =  nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Dropout(0.5)
#         )
#         self.msp_task_projection_head =  nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Dropout(0.5)
#         )

#         # Fully-connected layer
#         self.primary_task_classifier = nn.Sequential(
#             nn.Linear(fusion_feature.shape[1], n_class_primary)
#         )
#         self.vto_task_classifier = nn.Sequential(
#             nn.Linear(fusion_feature.shape[1], 9)
#         )
#         self.msp_task_classifier = nn.Sequential(
#             nn.Linear(fusion_feature.shape[1], 8)
#         )

#     def calculate_orthogonal_constraint(self, feature_1, feature_2):
#         assert feature_1.shape == feature_2.shape, "the dimension of two matrix is not equal"
#         N, C, H, W = feature_1.shape
#         feature_1, feature_2 = torch.reshape(feature_1, (N*C, H, W)), torch.reshape(feature_2, (N*C, H, W))
#         weight_squared = torch.bmm(feature_1, feature_2.permute(0, 2, 1))
#         # weight_squared = torch.norm(weight_squared, p=2)
#         ones = torch.ones(N*C, H, H, dtype=torch.float32).to(torch.device('cuda:0'))
#         diag = torch.eye(H, dtype=torch.float32).to(torch.device('cuda:0'))

#         loss = ((weight_squared * (ones - diag)) ** 2).sum()
#         return loss

#     def forward(self, x, task_name):
#         '''
#         @description: Complete the corresponding task according to the task tag
#         '''
#         # extract features

#         batch_size = len(x)
#         fea_shared_extract = torch.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2]))
#         fea_shared_extract = self.block_shared_feature_extractor_1(fea_shared_extract)
#         fea_shared_extract = fea_shared_extract.view(batch_size, fea_shared_extract.shape[3], fea_shared_extract.shape[2], fea_shared_extract.shape[1])
#         fea_shared_extract = self.block_shared_feature_extractor_2(fea_shared_extract)
#         fea_shared_extract = fea_shared_extract.view(batch_size, fea_shared_extract.shape[3], fea_shared_extract.shape[2], fea_shared_extract.shape[1])

#         # fea_after_fusion = self.block_feature_fusion(fea_shared_extract)

        
#         if task_name == "main":
#             # 推理过程
#             x = torch.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2]))
#             x = self.block_specific_main_feature_extractor_1(x)
#             x = x.view(batch_size, x.shape[3], x.shape[2], x.shape[1])
#             x = self.block_specific_main_feature_extractor_2(x)
#             fea_specific_main = x.view(batch_size, x.shape[3], x.shape[2], x.shape[1])

#             # print(fea_specific_main.shape, fea_shared_extract.shape)
#             # fea_main = fea_specific_main + fea_shared_extract
#             fea_main = torch.cat([fea_specific_main, fea_shared_extract], dim=1)

#             fea_main = self.block_feature_fusion(fea_main)
#             fea_main = self.main_task_projection_head(fea_main)
#             fea_main = fea_main.view(fea_main.size(0), -1)
#             logits_main = self.primary_task_classifier(fea_main)
#             pred_main = F.softmax(logits_main, dim = 1)
#             # 损失计算
#             orthogonal_constraint = self.calculate_orthogonal_constraint(fea_specific_main, fea_shared_extract)

#             return pred_main, orthogonal_constraint

#         elif task_name == "vto":
#             # 推理过程
#             x = torch.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2]))
#             x = self.block_specific_vto_feature_extractor_1(x)
#             x = x.view(batch_size, x.shape[3], x.shape[2], x.shape[1])
#             x = self.block_specific_vto_feature_extractor_2(x)
#             fea_specific_vto = x.view(batch_size, x.shape[3], x.shape[2], x.shape[1])
#             # fea_vto = fea_specific_vto + fea_shared_extract
#             fea_vto = torch.cat([fea_specific_vto, fea_shared_extract], dim=1)
#             fea_vto = self.block_feature_fusion(fea_vto)
#             fea_vto = self.vto_task_projection_head(fea_vto)

#             fea_vto = fea_vto.view(fea_vto.size(0), -1)
#             logits_vto = self.vto_task_classifier(fea_vto)
#             pred_vto = F.softmax(logits_vto, dim = 1)
#             # 损失计算
#             orthogonal_constraint = self.calculate_orthogonal_constraint(fea_specific_vto, fea_shared_extract)

#             return pred_vto, orthogonal_constraint

#         elif task_name == "msp":
#             # 推理过程
#             x = torch.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2]))
#             x = self.block_specific_msp_feature_extractor_1(x)
#             x = x.view(batch_size, x.shape[3], x.shape[2], x.shape[1])
#             x = self.block_specific_msp_feature_extractor_2(x)
#             fea_specific_msp = x.view(batch_size, x.shape[3], x.shape[2], x.shape[1])
#             # fea_msp = fea_specific_msp + fea_shared_extract
#             fea_msp = torch.cat([fea_specific_msp, fea_shared_extract], dim=1)
#             fea_msp = self.block_feature_fusion(fea_msp)
#             fea_msp = self.msp_task_projection_head(fea_msp)

#             fea_msp = fea_msp.view(fea_msp.size(0), -1)
#             logits_msp = self.msp_task_classifier(fea_msp)
#             pred_msp = F.softmax(logits_msp, dim = 1)
#             # 损失计算
#             orthogonal_constraint = self.calculate_orthogonal_constraint(fea_specific_msp, fea_shared_extract)

#             return pred_msp, orthogonal_constraint
#         else:
#             assert("TaskName Error!")

# class RSVPTransform(nn.Module):
#     """
#     Non-stationary Transformer
#     """
#     def __init__(self, configs):
#         super(RSVPTransform, self).__init__()
#         self.pred_len = configs.pred_len
#         self.seq_len = configs.seq_len
#         self.label_len = configs.label_len
#         self.output_attention = configs.output_attention

#         # Embedding
#         self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
#                                            configs.dropout)
#         self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
#                                            configs.dropout)
#         # Encoder
#         self.encoder = Encoder(
#             [
#                 EncoderLayer(
#                     AttentionLayer(
#                         DSAttention(False, configs.factor, attention_dropout=configs.dropout,
#                                       output_attention=configs.output_attention), configs.d_model, configs.n_heads),
#                     configs.d_model,
#                     configs.d_ff,
#                     dropout=configs.dropout,
#                     activation=configs.activation
#                 ) for l in range(configs.e_layers)
#             ],
#             norm_layer=torch.nn.LayerNorm(configs.d_model)
#         )
#         # Decoder
#         self.decoder = Decoder(
#             [
#                 DecoderLayer(
#                     AttentionLayer(
#                         DSAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
#                         configs.d_model, configs.n_heads),
#                     AttentionLayer(
#                         DSAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
#                         configs.d_model, configs.n_heads),
#                     configs.d_model,
#                     configs.d_ff,
#                     dropout=configs.dropout,
#                     activation=configs.activation,
#                 )
#                 for l in range(configs.d_layers)
#             ],
#             norm_layer=torch.nn.LayerNorm(configs.d_model),
#             projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
#         )

#         self.tau_learner   = Projector(enc_in=configs.enc_in, seq_len=configs.seq_len, hidden_dims=configs.p_hidden_dims, hidden_layers=configs.p_hidden_layers, output_dim=1)
#         self.delta_learner = Projector(enc_in=configs.enc_in, seq_len=configs.seq_len, hidden_dims=configs.p_hidden_dims, hidden_layers=configs.p_hidden_layers, output_dim=configs.seq_len)

#     def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
#                 enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

#         x_raw = x_enc.clone().detach()

#         # Normalization
#         mean_enc = x_enc.mean(1, keepdim=True).detach() # B x 1 x E
#         x_enc = x_enc - mean_enc
#         std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach() # B x 1 x E
#         x_enc = x_enc / std_enc
#         x_dec_new = torch.cat([x_enc[:, -self.label_len: , :], torch.zeros_like(x_dec[:, -self.pred_len:, :])], dim=1).to(x_enc.device).clone()

#         tau = self.tau_learner(x_raw, std_enc).exp()     # B x S x E, B x 1 x E -> B x 1, positive scalar    
#         delta = self.delta_learner(x_raw, mean_enc)      # B x S x E, B x 1 x E -> B x S

#         # Model Inference
#         enc_out = self.enc_embedding(x_enc, x_mark_enc)
#         enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask, tau=tau, delta=delta)

#         dec_out = self.dec_embedding(x_dec_new, x_mark_dec)
#         dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask, tau=tau, delta=delta)

#         # De-normalization
#         dec_out = dec_out * std_enc + mean_enc

#         if self.output_attention:
#             return dec_out[:, -self.pred_len:, :], attns
#         else:
#             return dec_out[:, -self.pred_len:, :]  # [B, L, D]
def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,    #dff,全卷积的维度
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,               #也就是说这里，TimesBlock输出的维度是d_model。那怎么把它再拼成一维？
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):           #因为nn.Module这个父类里有__call__调用了forward，所以在实例化TimesBlock时，会自动调用forward
                                    #并且，一旦继承了nn.Module类，给TimesBlock（x）传入x，会自动把x传给forward（self，x）中的x。所以我们在模型或layer实例化的时候没有看到参数接住x
        B, T, N = x.size()    
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:   #如果输入序列的长度+预测序列的长度 能被period整除
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,               #   B指block，
                              N).permute(0, 3, 1, 2).contiguous()          #contiguous()为了深拷贝
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)     #-1是通配符
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)            #res和perioad_weight都是四维的，它们的乘法，为什么不是矩阵相乘，而是直接在各ij坐标上相乘，为何不是标量积
        # residual connection
        res = res + x
        return res

class TimesNet(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(TimesNet, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)

        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)

    def classification(self, x_enc, x_mark_enc):
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.classification(x_enc, x_mark_enc)
        return dec_out  # [B, N]

if __name__ == "__main__":
    data = torch.tensor(np.random.rand(64, 64, 256)).to(torch.float32)
    model = FDN(n_class_primary=2, T = 256, channels=64, n_kernel_t=8, n_kernel_s=16, dropout=0.5, kernel_length=32)

    a = model(data, "main")

