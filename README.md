# MSPSNet-Change-Detection-TGRS
The Pytorch implementation for Deep Multiscale Siamese Network with Parallel Convolutional Structure and Self-Attention for Change Detection (MSPSNet)  
By Qingle Guo (e-mail:GQle_HIT@163.com), Junping Zhang, Shengyu Zhu and Chongxiao Zhong  
[10 Nov. 2021] Release the code of MSPSNet model  

##__Dataset Download__   
 LEVIR-CD：https://justchenhao.github.io/LEVIR/  
 SYSU：https://pan.baidu.com/share/init?surl=5lQPG_hXZbLp91VywwcT7Q　(mlls)  

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

@Article{
AUTHOR = {Qingle Guo, Junping Zhang and Ye Zhang},
TITLE = {Multitemporal Images Change Detection Based on AMMF and Spectral Constraint Strategy},
JOURNAL = {IEEE Transactions on Geoscience and Remote Sensing},
VOLUME = {59},
YEAR = {2021},
ISSUE = {4},
ARTICLE-NUMBER = {20965016},
URL = {https://ieeexplore.ieee.org/document/9143464},
ISSN = {1558-0644},
DOI = {10.1109/TGRS.2020.3008016}
}

```
__Acknowledgments__  

 Our code is inspired and revised by [pytorch-SNUNet], Thanks Sheng Fang for his great work!!  

__Reference__  
[1] H. Chen and Z. Shi, “A Spatial-temporal Attention-based Method and a New Dataset for Remote Sensing Image Change Detection,” Remote Sens., vol. 12, no. 10, pp. 1662, May 2020.  
[2] R. Daudt, B. Saux, and A. Boulch, “Fully Convolutional Siamese Networks for Change Detection,” in Proc. 25th IEEE Int. Conf. Image Process. (ICIP), pp. 4063–4067, Oct. 2018.  
[3] C. Zhang, P. Yue, D. Tapete, L. Jiang, and B. Shangguan, “A Deeply Supervised Image Fusion Network for Change Detection in High Resolution Bi-temporal Remote Sensing Images,” ISPRS J. Photogramm. Remote Sens., vol. 166, pp. 183–200, 2020.  
[4] J. Xu, C. Luo, X. Chen, S. Wei, and Y. Luo, “Remote Sensing Change Detection Based on Multidirectional Adaptive Feature Fusion and Perceptual Similarity,” Remote Sens., vol. 13, no. 15, pp. 3053-3077, Aug. 2021.  
