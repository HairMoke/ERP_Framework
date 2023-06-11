
# Author: Tammie li
# Description: 定义预处理方法 预处理方法是脑电信号处理的关键一环 必要时可以将其plot出来
# FilePath: \Utils\preprocess.py


from sklearn import preprocessing
from scipy import signal
import numpy as np
from scipy.linalg import sqrtm, inv
from collections import defaultdict
import mne
def bdf_to_numpy_2(bdf_data_path):

    bdf_file = bdf_data_path
    raw = mne.io.read_raw_bdf(bdf_file, preload=True)
    # raw.resample(256, npad='auto')#降采样
    # print(raw.info['sfreq'])
    fs = raw.info['sfreq']
    events = mne.find_events(raw, initial_event=True,min_duration=1 / 1024)#如果不降采样，这里应该是1/1024
    #min_duration 将事件通道中的更改视为事件所需的最短持续时间（以秒为单位）,因为biosemi发trigger过来会一连发好几个重复的
    raw.filter(0.5, 48)
    ch_names = raw.info['ch_names']
    print(ch_names)
    raw.pick_channels(ch_names[0:64])
    '''raw
    channel num : 64
    fs : 256Hz
    '''

    seq1 = mne.Epochs(raw, events, tmin=0, tmax=1,
                             baseline=(0, 0), reject_tmax=True).get_data()[:, :,1 ::4]#   N,C,T  N，通道，采样数据降采样4倍，从1024到256
    seq2 = mne.Epochs(raw, events, tmin=0, tmax=1,
                             baseline=(0, 0), reject_tmax=True).get_data()[:, :,1 ::4]#   N,C,T  N，通道，采样数据降采样4倍，从1024到256
    seq1 = np.delete(seq1,[i for i in range(seq1.shape[0]) if i%50 in [0]or (i+1)%50 in [0] or (i+2)%50 in [0]],axis=0)
    seq2 = np.delete(seq2,[i for i in range(seq2.shape[0]) if i%50 in [1,2,3]],axis=0)
    seq2 = np.delete(seq2,0,axis=0)
    
    
    # target_data = np.concatenate((seq1,seq2),axis=1)
    target_data = np.add(seq1 ,seq2)/2
    target = np.delete(events[:,-1], [i for i in range(events.shape[0]) if i%50 in [0]or (i+1)%50 in [0] or (i+2)%50 in [0]])

    idx = np.where(target==4)
    target[idx] = 16    #把“非+阴”、“阳+非”的情况都当成非目标，弄成了两分类

    return target, target_data


class DataProcess:
    def __init__(self):
        pass
    
    def scale_data(self, data):
        # 归一化数据 data = [N, C, T]
        scaler = preprocessing.StandardScaler()
        for i in range(data.shape[0]):
            data[i, :, :] = scaler.fit_transform(data[i, :, :])
        return data

    def band_pass_filter(self, data, freq_low, freq_high, fs):
        # 带通滤波
        wn = [freq_low * 2 / fs, freq_high * 2 /fs]
        b, a = signal.butter(3, wn, 'bandpass')
        for sample in range(data.shape[0]):
            for channel in range(data.shape[1]):
                data[sample, channel, ...] = signal.filtfilt(b, a, data[sample, channel, ...])
        return data

    def euclidean_space_alignment(self, data):
        """Transfer Learning for Brain–Computer Interfaces: A Euclidean Space Data Alignment Approach"""
        # data->(N, C, T), 需要先执行滤波操作
        # 公式10-计算协方差
        r = 0
        for trial in data:
            cov = np.cov(trial, rowvar=True)
            r += cov
        r = r/data.shape[0]
        # 公式11
        r_op = inv(sqrtm(r))

        results = np.matmul(r_op, data)
        return results

if __name__ == "__main__":
    data = np.random.rand(10, 64, 256)
    prePocessor = DataProcess()
    result = prePocessor.euclidean_space_alignment(data)



