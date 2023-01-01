# -*- coding: utf-8 -*- #
'''
--------------------------------------------------------------------------
# File Name:        PATH_ROOT/config.py
# Author:           JunJie Ren
# Version:          v2.0
# Created:          2022/04/01
# Description:      — — — — — — — — — — — — — — — — — — — — — — — — — — — 
                              --> 电磁扰动识别分类训练程序 <--        
                    -- 参数配置文件
                    — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Module called:    None
# Function List:    None
# Class List:       None
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# History:
       |  <author>  | <version> |   <time>   |          <desc>
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
   <0> | JunJie Ren |   v1.0    | 2022/04/01 |          creat
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
   <0> | JunJie Ren |   v1.2    | 2022/11/24 |       完善相关参数
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
   <3> | JunJie Ren |   v2.0    | 2023/01/01 |      适配域适应功能
# ------------------------------------------------------------------------
'''

class Configs(object):
   def __init__(self):
      
      ''' 数据集划分参数配置 '''
      self.divide_dataset_name = "UAV2022.11.18_dtmb04_0.01"  # 此次数据集划分名称
      self.divide_dataset_savePath = "./dataset"      # 此次数据集划分保存的路径
      self.classes = ["DH","DQ","DS","DX","DY","DZ"]                     # 数据集类别
      self.reshape_size = (224, 224)                   # 数据重塑尺寸
      self.train_divide = {
         "data_path": "/data/UAV_DATA_2022.11.20/2022-11-18_tdms/DTMB_04/Train",      # 训练集数据路径
         "crop_len": self.reshape_size[0]*self.reshape_size[1],          # 训练集裁剪数据路径
         "conti_Len": 0,
         "trainset_ratio": 0.01,
         "testset_ratio": 0,
      }
      self.test_divide = {
         "data_path": "/data/UAV_DATA_2022.11.20/2022-11-18_tdms/DTMB_04/Test",      # 训练集数据路径
         "crop_len": self.reshape_size[0]*self.reshape_size[1],          # 训练集裁剪数据路径
         "conti_Len": 0,
         "trainset_ratio": 0,
         "testset_ratio": 0.01,
      }


      ''' 训练参数配置 '''
      # model
      self.model = "ResNet50"                          # 指定模型，ResNet50,ResNet101,ResNet152,ResNet_50_SE,UAV_CA_SA
      self.exp_num = 1                                # 实验次数
      self.note = "transfer_107per_base"                                    # 实验备注
      self.gup_id = "2"                               # 指定GPU
      
      self.resume = True                               # 是否加载训练好的模型
      self.resume_path = "./work_dir/UAV2022.11.16_0.5_ResNet50_x30_6c_1ch_no1_/UAV2022.11.16_0.5_ResNet50_x30_6c_1ch_no1_.pth" # 加载模型路径

      # Dataset
      self.dataset = "UAV2022.11.18_dtmb04_0.01"          
      self.channels = ["CH0",]     # 数据集通道
      self.process_IQ = True                           # 是否在载入数据时对IQ两路进行预处理
      self.TRAIN_PATH = "./dataset/"+self.dataset+"/train_set.txt"
      self.VALID_PATH = "./dataset/"+self.dataset+"/test_set.txt"
      self.VALID_PATH = "./dataset/UAV2022.11.18_dtmb04_0.5/test_set.txt"
      self.FINAL_TEST_PATH = "./dataset/UAV2022.11.18_dtmb04_0.5/test_set.txt"

      self.add_noise = 0                                 # 是否添加噪声, 0:不添加, 1:添加, 0-1:按比例随机添加
      self.SNR_min = 10                                   # 噪声最小信噪比
      self.SNR_max = 20                                   # 噪声最大信噪比

      # train
      self.batch_size = 32                              # DataLoader中batch大小，550/110=5 Iter
      self.num_workers = 16                             # DataLoader中的多线程数量
      self.num_epochs = 300                              # 训练轮数
      self.lr = 0.001                                   # 初始lr
      self.valid_freq = 1                              # 每几个epoch验证一次

      # log
      self.iter_smooth = 10                             # 打印 & 记录log的频率,每几个batch打印一次准确率
      self.checkpoint_name = self.dataset+"_"+self.model+"_x"+str(self.num_epochs)+"_"+str(len(self.classes))+"c_"+str(len(self.channels))+"ch_no"+str(self.exp_num)+"_"+self.note


   def get_members(self):
        return vars(self)