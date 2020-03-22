# tf-faster-rcnn-ghostnet
__This is a demo about Faster R-CNN use GhostNet__


## Explain
This is a demo of Faster R-CNN for Windows. The backbone network uses GhostNet.  

Faster R-CNN code from [dBeker](https://github.com/dBeker/Faster-RCNN-TensorFlow-Python3)  

GhostNet code from [huawei-noah](https://github.com/huawei-noah/ghostnet)  

This code is only for learning and communication, it is not certain that the code is completely correct. Due to limited computing resources, no suitable hyperparameters were found, and results did not perform well.

## Dependence
python 3.6  
tensorflow 1.8.0  
tensorpack 0.9.7

## How To Use
You need to make a dataset in VOC format   

If it doesn't work, you need to set it up [here](https://github.com/dBeker/Faster-RCNN-TensorFlow-Python3) and replace it with the corresponding file in the library  

　__1. Put the dataset into './data'__  

　__2. in './lib/datasets/pascal_voc.py' to change class__  
 
　`<self.path = r'D:\Faster-R-CNN\labels_13_BJOYYX.txt'>`  

　__3. Set pre-training file path__ 
 
　`<pretrained_model = r'F:\GhostNet\ghostnet\models\ghostnet_checkpoint'>`
 
run train-ghostnet.py to train  

run test-ghostnet.py to test  

## Results
In my own dataset, batch=128, iteration=100000, results:

![result](https://github.com/S0soo/tf-faster-rcnn-ghostnet/blob/master/output/000548.jpg)  

The results are not ideal, and I will continue to search for more suitable hyperparameters.   
GhostNet is tiny.





