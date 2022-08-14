# Hubmap-segmentation-competition      

Since the image resolution is quite large,I decide to use iafoss's trick to divide image into titles.By doing this,i can create small dataset without losing information from original image.I use this notebook to do this step        
      
Preprocessing     
|__converting-to-256x256.ipynb      

dataset :https://www.kaggle.com/datasets/bhuynguyn/hubmaphacking2022-comp-256x256     
       
After creating new dataset, I create data training pipeline:    
I applied some data augmentation techniques such as CLAHE, RandomBrightness, RandomContrast,… to                  
enhance the quality of images and enrich training data.       
data_prepare        
|__dataloader.py       

Model       
Ensemble model : U-Resnet34+SCSE+FPA+Hypercolumns           

![image](https://user-images.githubusercontent.com/90911918/184531888-7e40b157-6482-42bf-b0dd-53be2fde74ef.png)      
link :https://arxiv.org/pdf/1904.04445.pdf       

Authors inserted SCSE modules after each encoder and decoder blocks.They act as attention module,using spatial attention mechanism helps model to focus on important features and suppress less important ones.      

Additionally, in the bottleneck block between the encoder and the decoder authors use Feature Pyramid Attention module, which increases the receptive field by fusing features from different pyramid scales.By doing this, model will be able to focus on object at different places in the picture.           

To exploiting feature maps from different scales authors use Hypercolumns. They stack the upsampled feature maps
from all decoder blocks and use them as the input to the final layer. Therefore,they could get more precise localization and captures the semantics at the same time.         

Pranet model     
![image](https://user-images.githubusercontent.com/90911918/184532637-f4b79b2c-8fbe-4b08-afd4-9e77a7354111.png)      

In this architecture, backbone resnet50 is used for extracting feature maps from different convolution layers.Feature maps after each convolution layer are fed into RFB(receptive field block) .RFB block is designed with 4 branches,using diliate convolution layers with different rate.capture global contextual information of object in the picture,             


