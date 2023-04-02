import skvideo.io
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.autograd import Variable
from skimage.metrics import structural_similarity as ssim
import scipy
import argparse


weight=[1/30,1,5,1]
non_uni=[0.367,0.633]



if __name__ == "__main__":    
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "-n", "--name", type=str, 
        default="example.mp4", 
        help="input video path",
    )
    
    parser.add_argument(
        "-i", "--info", type=str, 
        default="example.csv", 
        help="video information"
    )

    parser.add_argument(
        "-q", "--qos", type=str, 
        default="height", 
        help="QoS feature representing quality"
    )
    
    parser.add_argument(
        "-d", "--dir", type=str, 
        default="example.mat", 
        help="feature save dictionary"
    )

    parser.add_argument(
        "-s", "--sample", type=int, 
        default=30, 
        help="global sampling intensity"
    )

    
    args = parser.parse_args()
    videoName=args.name
    videoIndex=args.info
    dirSave=args.dir
    QoS=args.qos
    
    videoData = skvideo.io.vread(videoName,as_grey=True)
    videoInfo = pd.read_csv(videoIndex)
    
    start_index=np.zeros(5,dtype='int')
    end_index=np.zeros(5,dtype='int')
    start_index[0]=int(np.round(videoInfo['framerate'][0]*videoInfo['rebuffering_duration'][0]))
    end_index[0]=start_index[0]+int(np.round(videoInfo['framerate'][0]*videoInfo['chunk_duration'][0]))
    
    for clips in range(1,5):
        start_index[clips]=end_index[clips-1]+int(np.round(videoInfo['framerate'][clips]*videoInfo['rebuffering_duration'][clips]))
        end_index[clips]=start_index[clips]+int(np.round(videoInfo['framerate'][clips]*videoInfo['chunk_duration'][clips]))
    reward_QoS=np.zeros(5)
    reward_content=np.zeros(25)
    penalty_QoS=np.zeros(2)
    penalty_content=np.zeros(4)
    
    ##################### reward QoS #####################
    ######################################################
    for clips in range(0,5):
        reward_QoS[clips]=videoInfo[QoS][clips]
    ######################################################

    
    ##################### reward content #################
    ######################################################
    gru = nn.GRU(input_size=4, hidden_size=4,
                 num_layers=2,  # gru层数
                 batch_first=False,  # 默认参数 True:(batch, seq, feature) False：True:( seq,batch, feature),
                 bidirectional=False,  # 默认参数
                 )
    #hid = torch.randn(2 * 1, 1, 4)
    hid = torch.Tensor([[[ 0.3818, -0.0660, -0.0082, -0.4698]],[[-0.2710,  0.1756,  0.3829,  0.4570]]])

    for clips in range(0,5):
        start_frame=start_index[clips]
        end_frame=end_index[clips]
        resnet50_feature_extractor = models.resnet50(pretrained=True)  # 导入ResNet50的预训练模型
        resnet50_feature_extractor.conv1=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  # 黑白
        resnet50_feature_extractor.fc = nn.Linear(2048, 16)  # 最后16个元素
        #torch.nn.init.eye(resnet50_feature_extractor.fc.weight)  # 将二维tensor初始化为单位矩阵
        transform1 = transforms.Compose([  
            transforms.Scale(256),  # 缩放
            transforms.CenterCrop(224),  # 中心裁剪
            transforms.ToTensor()]  # 转换成Tensor
        )
        now_frame=start_frame
        while now_frame<end_frame:
            x = Variable(torch.unsqueeze(transform1(transforms.ToPILImage()(videoData[now_frame,:,:,:])), dim=0).float(), requires_grad=False)
            y = resnet50_feature_extractor(x)
            y = y.data.numpy()
            inter_frame=args.sample
            if now_frame*2<(start_frame+end_frame):
                inter_frame=inter_frame*non_uni[0]
            else:
                inter_frame=inter_frame*non_uni[1]
            now_frame=now_frame+int(inter_frame*len(non_uni))
            output, hid = gru(torch.Tensor([[[np.max(y),np.min(y),np.mean(y),np.std(y)]]]), hid)
        reward_content[clips*5:(clips*5+4)]=output.data.numpy()[0][0]
        imgComplex=x.data.numpy()[0,0,:,:]
        tmp=np.zeros([16,16])
        ave=scipy.linalg.block_diag(tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp)/16
        mat1=ave*imgComplex
        mat2=imgComplex*ave
        mat3=(mat1+mat2)/2
        reward_content[clips*5+4]=min(np.mean(abs(imgComplex-mat1)),np.mean(abs(imgComplex-mat2)),np.mean(abs(imgComplex-mat3)))     
    ######################################################
    
    
    ##################### penalty QoS ####################
    ######################################################
    penalty_QoS[0]=videoInfo['rebuffering_duration'][0]
    penalty_QoS[1]=np.mean(videoInfo['rebuffering_duration'][1:5])
    ######################################################
    
    ##################### penalty content ################
    ######################################################
    c1=0.5
    c2=100
    for clips in range(0,4):
        rebuffer=videoInfo['rebuffering_duration'][clips]
        switch=max(videoInfo[QoS][clips]-videoInfo[QoS][clips+1],0)
        structure=ssim(videoData[end_index[clips]-5,:,:,0],videoData[start_index[clips+1]+5,:,:,0])
        penalty_content[clips]=(1+rebuffer/c1)*(1+switch/c2)*structure
    ######################################################    
    
    scipy.io.savemat(dirSave, mdict={'all': np.hstack((reward_QoS*weight[0], reward_content*weight[1], penalty_QoS*weight[2], penalty_content*weight[3]))})
    print("The feature of " + videoName + " is saved")
