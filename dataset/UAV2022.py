# -*- coding: utf-8 -*- #
'''
--------------------------------------------------------------------------
# File Name:        PATH_ROOT/dataset/UAV2022.py
# Author:           JunJie Ren
# Version:          v3.0
# Created:          2022/04/01
# Description:      — — — — — — — — — — — — — — — — — — — — — — — — — — — 
                            --> UAV电磁扰动识别代码 (PyTorch) <--        
                    -- 数据集处理载入程序
                    -- 要有一个train.txt/test.txt
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Module called:    <0> None
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Function List:    <0> read_txt(): 
                        -- 从.txt中读取训练,测试样本的标签,路径,采样点信息
                    <1> divide_dataset():
                        -- 划分数据集,生成划分好的.txt
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Class List:       <0> UAVDataset(Dataset): 
                        -- 定义UAVDataset类,继承Dataset方法,并重写
                        __getitem__()和__len__()方法
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# History:
       |  <author>  | <version> |   <time>   |          <desc>
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
   <0> | JunJie Ren |   v1.0    | 2022/04/01 |   完成初步数据载入功能
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
   <1> | JunJie Ren |   v2.0    | 2022/11/24 | 直接从tdms文件中读取数据
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
   <2> | JunJie Ren |   v2.1    | 2022/11/24 |    优化配置文件载入逻辑
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
   <3> | JunJie Ren |   v3.0    | 2023/01/01 |    新增域适应功能支持
--------------------------------------------------------------------------
'''

import os
import sys
import random
import argparse
import importlib

import numpy as np

from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

class UAVDataset(Dataset):
    def __init__(
            self, 
            txt_path, 
            channels=["CH0"],
            reshape_size=None, 
            processIQ=True, 
            return_label=True, 
            transform=None, 
            add_noise=False, SNR_min=0, SNR_max=10,
        ):
        # 标签,路径,采样点起始位置,信号采样长度
        self.labels, self.sigs_path, self.start_poses, self.sigs_len = read_txt(txt_path)
        self.transform = transform
        self.channels = channels
        self.size = reshape_size        # 自定义输出大小,(a,b,...)
        self.processIQ = processIQ
        self.return_label = return_label

        self.add_noise = add_noise
        self.SNR_min = SNR_min
        self.SNR_max = SNR_max

    def __getitem__(self, index):
        label = self.labels[index]
        sig = np.zeros((self.sigs_len[index], len(self.channels)*2), dtype=np.float32)
        for i in range(len(self.channels)):
            sig_path = self.sigs_path[index].replace("CH0", self.channels[i])
            sig_CH = tdms2Array(
                tdms_file_path=sig_path,
                start_pos=self.start_poses[index],
                sample_point_len=self.sigs_len[index],
            )
            sig[:, i*2:i*2+2] = sig_CH
        # 使用awgn添加噪声,并按照一定的几率添加噪声
        if random.random() < self.add_noise:
            SNRdB = random.randint(self.SNR_min, self.SNR_max)
            # print("SNRdB: ", SNRdB)
            sig = awgn(sig, SNRdB)

        # 归一化
        if self.processIQ:
            sig = processIQ(sig)
        sig = (sig.T)[:,:]

        # 重塑输入大小
        if self.size is not None:
            sig = np.reshape(sig, (len(self.channels)*2, self.size[0], self.size[1]))

        # 数据增强
        if self.transform is not None:
            sig = self.transform(sig)

        if self.return_label:
            return sig, label
        else:
            return sig
  
    def __len__(self):
        return len(self.labels)


def read_txt(path):
    labels, sigs_path, start_poses, sigs_len = [], [], [], []
    with open(path, 'r') as f:
        for line in f.readlines():
            label, sig_path, start_pos, sig_len  = line.strip().split(',')

            labels.append(int(label))
            sigs_path.append(sig_path)
            start_poses.append(int(start_pos))
            sigs_len.append(int(sig_len))
            
    return labels, sigs_path, start_poses, sigs_len


def get_tdms_len(tdmsPath, offset = 4096):
    """ 
    Funcs:
        快速获取.tdms文件(双通道,N*2)的采样点个数
    Args:
        <0> tdmsPath: str
        <1> offset: tdms文件正式数据偏移量
    Returns:
        <0> sampleLen: int, 采样点数
    """
    with open(tdmsPath, 'rb') as f_tdms:
        f_tdms.seek(0,2)            #指针移至末尾
        # 当前指针-4096)//4计算可读采样点数
        sampleLen = (f_tdms.tell()-offset)//4 
        return sampleLen


def tdms2Array(tdms_file_path, start_pos=0, sample_point_len=50000, offset=4096):
    '''
    Funcs:
        从.tdms文件中快速读取数据，并转换为numpy数组
    Args:
        <0> tdms_file_path :        tdms文件路径
        <1> offset :                tdms文件正式数据偏移量,默认4096,该部分用于存储文件信息,不包含真实采样点信息
        <2> sample_point_len :      采样点长度，最终会得到(sample_point_len, 2)的numpy数组
        <3> ingonor_sample_point :  要跳过的采样点个数(一个采样点个数 = 一个时间点两个通道的数据)
    '''
    with open(tdms_file_path, 'rb') as tdms_file:
        tdms_file.seek(0,2)  #指针移至末尾
        total_sample_point_capacity = (tdms_file.tell()-offset)//4 #(当前指针-4096)//4计算可读采样点数

        if start_pos+sample_point_len > total_sample_point_capacity: #判断可用采样点数是否满足要求
            print('可用{}个采样点不足{},本次读取所有可用采样点'.format(total_sample_point_capacity-start_pos, sample_point_len))

        tdms_file.seek(offset+start_pos*4,0) #指针移至要采样点处
        byte_stream = tdms_file.read(4*sample_point_len) #成对(双通道)读取数据
        tdms2array = np.frombuffer(byte_stream,dtype=np.int16,offset=0).reshape((sample_point_len,2))  #字节数据流转numpy
        return tdms2array


def processIQ(x):
    ''' 对N路信号分别进行预处理,结合两路为复数,除以标准差,再分离实部虚部到两路 '''
    y = np.zeros_like(x, dtype=np.float32)
    for i in range(x.shape[1]//2):
        sample_complex = x[:, 2*i] + x[:, 2*i+1] * 1j
        sample_complex = sample_complex / np.std(sample_complex)
        # sample_complex -= np.min(sample_complex)
        # sample_complex /= np.max(sample_complex)
        y[:, 2*i] = sample_complex.real
        y[:, 2*i+1] = sample_complex.imag
    return y


def divide_dataset(dataset_name, data_path, save_path, crop_len = 224*224,
                    classes = ["N","Y","D"], channels = ["CH0"], conti_Len=0, 
                    trainset_ratio = 0.5, testset_ratio = 1):
    """ 
    Funcs:
        根据.tdms文件路径与长度,划分数据集
    Args:
        <0> dataset_name: str, 此次划分的数据集自定义名称
        <1> data_path: str, 原始数据存放路径
        <2> save_path: str, 数据集索引保存路径
        <4> crop_len: int, 信号裁剪长度, 最后一段不足crop_len的丢弃

        <5> channels: list, 使用的信号通道, 默认只["CH0"]
        <7> conti_Len: int, 划分时连续信号长度, 默认0, 即不连续, 随机划分样本

        <5> trainset_ratio: float, 训练数据占总数据的比例
        <6> testset_ratio: float, 除去训练数据外,测试数据占剩余数据的比例
    Returns:
        <0> None
    """
    save_dir = (f"{save_path}/{dataset_name}")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if trainset_ratio != 0:
        train_txt = open(f'{save_path}/{dataset_name}/train_set.txt', 'w')
    if testset_ratio != 0:
        test_txt = open(f'{save_path}/{dataset_name}/test_set.txt', 'w')

    for dir in os.listdir(data_path):
        # 遍历类别文件夹
        if dir in classes:
            num_tdms = 0
            # 只选中指定类别的文件夹
            label = np.where(np.array(classes) == dir)[0][0]  # 该文件夹下的类别
            dir_path = os.path.join(data_path, dir)
            for file in os.listdir(dir_path):
                # 遍历该类别下的.tdms文件
                if os.path.splitext(file)[1] == '.tdms':
                    if os.path.splitext(file)[0].split('_')[-1] not in channels:
                        continue
                    tdms_path = os.path.join(dir_path, file)    # .tdms文件路径
                    crop_num = (get_tdms_len(tdms_path) // crop_len)-1   # 一个.tdms裁剪后的样本数量

                    # 正常情况下，按一定比例划分数据集
                    train_idx = random.sample(range(crop_num), int(crop_num*trainset_ratio))
                    test_idx = list(set(range(crop_num)).difference(set(train_idx)))
                    test_idx = random.sample(test_idx, int(len(test_idx)*testset_ratio))

                    # 如果指定了连续长度，将按固定连续长度划分数据集
                    if trainset_ratio == 1 and conti_Len > 0:
                        start_idx = random.randint(1, crop_num-conti_Len)
                        train_idx = np.arange(start_idx, start_idx+conti_Len)
                    if testset_ratio == 1 and conti_Len > 0:
                        start_idx = random.randint(1, crop_num-conti_Len)
                        test_idx = np.arange(start_idx, start_idx+conti_Len)

                    # 保存至.txt
                    for idx in train_idx:
                        train_txt.write(f"{label},{tdms_path},{idx*crop_len},{crop_len}\n")
                    for idx in test_idx:
                        test_txt.write(f"{label},{tdms_path},{idx*crop_len},{crop_len}\n")
                    num_tdms += 1


# author - Mathuranathan Viswanathan (gaussianwaves.com
# This code is part of the book Digital Modulations using Python

from numpy import sum,isrealobj,sqrt
from numpy.random import standard_normal

def awgn(s,SNRdB,L=1):
    """
    AWGN channel
    Add AWGN noise to input signal. The function adds AWGN noise vector to signal 's' to generate a resulting signal vector 'r' of specified SNR in dB. It also
    returns the noise vector 'n' that is added to the signal 's' and the power spectral density N0 of noise added
    Parameters:
        s : input/transmitted signal vector
        SNRdB : desired signal to noise ratio (expressed in dB) for the received signal
        L : oversampling factor (applicable for waveform simulation) default L = 1.
    Returns:
        r : received signal vector (r=s+n)
    """
    gamma = 10**(SNRdB/10) #SNR to linear scale
    if s.ndim==1:# if s is single dimensional vector
        P=L*sum(abs(s)**2)/len(s) #Actual power in the vector
    else: # multi-dimensional signals like MFSK
        P=L*sum(sum(abs(s)**2))/len(s) # if s is a matrix [MxN]
    N0=P/gamma # Find the noise spectral density
    if isrealobj(s):# check if input is real/complex object type
        n = sqrt(N0/2)*standard_normal(s.shape) # computed noise
    else:
        n = sqrt(N0/2)*(standard_normal(s.shape)+1j*standard_normal(s.shape))
    r = s + n # received signal
    return r


if __name__ == "__main__":
    ''' 划分数据集 / 测试UAV2022.py, 测试dataLoader是否正常读取、处理数据 '''
    parser = argparse.ArgumentParser(
        description='Divide/test the dataset'
    )
    parser.add_argument(
        '--config', 
        default="/data1/jjren/UAV_Detection_wav/configs/default_configs.py",
        type=str,
        help='Configuration file Path'
    )
    args = parser.parse_args()

    # 动态加载配置文件
    sys.path.append(os.path.dirname(args.config))
    module_name = os.path.splitext(os.path.basename(args.config))[0]
    cfgs = importlib.import_module(module_name).Configs()
    
    # 数据集的划分
    divide_dataset(     # train
        cfgs.divide_dataset_name, 
        data_path=cfgs.train_divide["data_path"], 
        save_path=cfgs.divide_dataset_savePath, 
        classes=cfgs.classes,
        crop_len=cfgs.train_divide["crop_len"],
        conti_Len=cfgs.train_divide["conti_Len"],
        trainset_ratio=cfgs.train_divide["trainset_ratio"],
        testset_ratio=cfgs.train_divide["testset_ratio"]
    )
    divide_dataset(     # valid
        cfgs.divide_dataset_name, 
        data_path=cfgs.test_divide["data_path"], 
        save_path=cfgs.divide_dataset_savePath, 
        classes=cfgs.classes,
        crop_len=cfgs.test_divide["crop_len"],
        conti_Len=cfgs.test_divide["conti_Len"],
        trainset_ratio=cfgs.test_divide["trainset_ratio"],
        testset_ratio=cfgs.test_divide["testset_ratio"]
    )

    # 数据集测试
    # transform = transforms.Compose([ 
    #     # transforms.ToTensor()
    #     # waiting add
    # ])
    # train_dataset = UAVDataset(
    #     cfgs.TRAIN_PATH,
    #     channels=cfgs.channels, 
    #     reshape_size=cfgs.reshape_size, 
    #     transform=transform,
    #     processIQ=cfgs.process_IQ, 
    #     return_path=True,
    #     add_noise=cfgs.add_noise,
    #     SNR_min=cfgs.SNR_min,
    #     SNR_max=cfgs.SNR_max,
    # )
    # # 通过DataLoader读取数据
    # train_loader = DataLoader( 
    #     train_dataset, 
    #     batch_size = cfgs.batch_size, 
    #     num_workers = cfgs.num_workers, 
    #     shuffle = True, 
    #     drop_last = False
    # )

    # from tqdm import tqdm
    # for datas,targets,paths in tqdm(train_loader):
    #     for data,target,path in zip(datas,targets,paths):
    #         print(data.shape,target,path)
    #     pass

