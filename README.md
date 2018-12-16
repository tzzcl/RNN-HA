# RNN-HA

This is the offical website for the implementation of RNN-HA. 

This package is developed by Mr. Chen-Lin Zhang (http://lamda.nju.edu.cn/zhangcl/) and Dr. Xiu-Shen Wei (http://lamda.nju.edu.cn/weixs/). If you have any problem about 
the code, please feel free to contact Mr. Chen-Lin Zhang (zhangcl@lamda.nju.edu.cn). 
The package is free for academic usage. You can run it at your own risk. For other purposes, please contact Dr. Xiu-Shen Wei (weixs.gm@gmail.com).

If you find our package is useful to your research. Please cite our paper:

Reference: 
           
[1] X.-S. Wei, C.-L. Zhang, L.-Q. Liu, J. Wu and C. Shen. Coarse-to-fine: A RNN-based hierarchical attention model for vehicle re-identification. In Proceedings of the 14th Asian Conference on Computer Vision (ACCV 2018), Perth, Australia, in press.
## Requirements
The code needs PyTorch(https://pytorch.org/).

More specifically, the experiments is conducted under PyTorch v0.3.1 with CUDA 8.0. (For PyTorch version > 0.4.0, it will result in a GPU memory error). 


## Usage

For training RNN-HA, there are some following steps:

### prepare datasets and fine-tune base models:
For VehicleID and VeRi datasets, you need to first get them by yourself:

Please follow the instructions in (https://pkuml.org/resources/pku-vehicleid.html) and (https://github.com/VehicleReId/VeRidataset).

After you get these datasets, you need to do the following steps:

First you need to prepare torch style training images. We prepare a Matlab script `prepare_torch.m` to conduct the task.

For VGG_M model conversion, we follow the guide in (https://github.com/fanq15/caffe_to_torch_to_pytorch).
After preparing training images and baseline models, you can use `fine_tune.m` to fine tune the baseline model on specific datasets.

### Training

For training RNN-HA, you first need to change some settings in `main_GRU_fusion_attention.py`, and you can simply run `main_GRU_fusion_attention.py`

### Testing

For testing RNN-HA, you first need to change some settings in `main_GRU_fusion_attention_test.py`, and you can simply run `main_GRU_fusion_attention_test.py`

And you will get the final feature matrix for evaulation.
