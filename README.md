# MSPSNet-Change-Detection-TGRS
The Pytorch implementation for Deep Multiscale Siamese Network with Parallel Convolutional Structure and Self-Attention for Change Detection  
Qingle Guo, Junping Zhang, Shengyu Zhu and Chongxiao Zhong  
[04 Dec. 2021] Release the first version of the MSPSNet

__Dataset Download__   
 LEVIR-CD：https://justchenhao.github.io/LEVIR/  
 SYSU：https://drive.google.com/drive/folders/1ALb8rzw9zEMSxwNTvIrIaA83zjjs04CE  

 Note: Please crop the LEVIR dataset to a slice of 256×256 before training with it.  

__Dataset Path Setteing__  
 LEVIR_CD or SYSU  
     |—train  
          |   |—A  
          |   |—B  
          |   |—OUT  
     |—val  
          |   |—A  
          |   |—B  
          |   |—OUT  
     |—test  
          |   |—A  
          |   |—B  
          |   |—OUT  
 Where A contains images of first temporal image, B contains images of second temporal images, and OUT contains groundtruth maps.  

__Traing and test Process__   

 python train.py  
 python test.py  

__Revised parameters__  
 You can revised related parameters in the "metadata.json" file.  

__Requirement__  

-Pytorch 1.8.0  
-torchvision 0.9.0  
-python 3.8  
-opencv-python  4.5.3.56  
-tensorboardx 2.4  
-Cuda 11.3.1  
-Cudnn 11.3  


__Citation__  

 If you use this code for your research, please cite our papers.  

```
@Article{  
AUTHOR = {Qingle, Guo, Junping Zhang, Shengyu Zhu, Chongxiao Zhong and Ye Zhang},  
TITLE = {Deep Multiscale Siamese Network with Parallel Convolutional Structure and Self-Attention for Change Detection },
JOURNAL = {IEEE Transactions on Geoscience and Remote Sensing},
VOLUME = {},
YEAR = {2022},
ISSN = {1558-0644},
}

```
__Acknowledgments__  

 Our code is inspired and revised by [pytorch-SNUNet], Thanks Sheng Fang for his great work!!  

__Reference__  
[1] H. Chen and Z. Shi, “A Spatial-temporal Attention-based Method and a New Dataset for Remote Sensing Image Change Detection,” Remote Sens., vol. 12, no. 10, pp. 1662, May 2020.  
[2] S. Fang, K. Li, J. Shao and Z. Li, “SNUNet-CD: A Densely Connected Siamese Network for Change Detection of VHR Images,” IEEE Geosci.and Remote Sens. Lett., 2021.  
[3] Q. Shi, M. Liu, S. Li, X. Liu, F. Wang, L. Zhang, "A Deeply Supervised Attention Metric-Based Network and an Open Aerial Image Dataset for Remote Sensing Change Detection," IEEE Trans. Geosci. Remote Sens., 2021
