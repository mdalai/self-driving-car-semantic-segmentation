# self-driving-car-semantic-segmentation

[//]: # (Image References)
[train_label1]: ./assets/train_label1.PNG
[train_label2]: ./assets/train_label2.PNG
[FCN_arch]: ./assets/FNC_architecture.PNG
[round11]: ./assets/round11.PNG
[round12]: ./assets/round12.PNG
[round13]: ./assets/round13.PNG
[round21]: ./assets/round21.PNG
[round22]: ./assets/round22.PNG
[round23]: ./assets/round23.PNG
[lr0009]: ./assets/lr0009.PNG
[lr0001]: ./assets/lr0001.PNG
[lr00001]: ./assets/lr00001.PNG
[round31]: ./assets/round31.PNG
[round32]: ./assets/round32.PNG
[round33]: ./assets/round33.PNG


## Project Introduction
**Project Goal**: classify each pixel in road images into Road or Not Road.

**Project Overview**:
We will use [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) to train a model called [Fully Convolutional Network (FCN)](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf), then using the trained FCN model to classify pixels in test images.

## Dataset Understanding
Download the dataset in [HERE](http://www.cvlibs.net/download.php?file=data_road.zip).

Let's see what training image and label look like:

![alt text][train_label1]
![alt text][train_label2]

- Road: labeled as PINK.
- Not Road: labeled as RED.
- Image shape:
    - Train image shape:  (375, 1242, 3).
    - Label image shape:  (375, 1242, 3).
- Training Data size: 
  - Total training images:  289.
  - Total training lables:  289.

## Fully Convolutional Network (FCN)
Typical CNN can classify Hot Dog and NOT Hot Dog from image very well, but it can not answer the question of where is the hot dog in the image. Because it does not preserve spatial information from the image. FCN solves this problem.

![alt text][FCN_arch]

Fully Convolutional Networks (FCN) consists of two parts: Encoder and Decoder.

**Encoder**: extract features from the image.
- Use state of art pre-trained model such as VGG, ResNet. These are well-trained models performed great in ImageNet.
- Replace the final fully-connected layer with 1x1 Convolution. 1x1 Convolution makes it possible to preserve spatial information.

**Decoder**: upscale the output from Encoder make it same size as original image.
- Upsample the layers. Transpose convolution help upsampling the previous layer to a higher resolution or dimension. Transpose convolution is also called deconvolution. It is opposite of convolution process.
- Skip layers. Problem of convolution in encoder is that it looks close on some feature and lose bigger picture as a result. The skip layer is here to retain losed information. Simply, it skipes some of the layers in the encoder and decoder layers. 


## Training & Testing

### AWS Setup
In order to make training process FUN, we want to use GPU. But setting up GPU instance on AWS ends up Not FUN for me. 

**AWS g2.2xlarge  instance type won’t work for Udacity AMI**

It took me LONG time try setting up AWS for the project. I tried all following works back and forth:
- Error: [Illegal instruction (core dumped)](https://discussions.udacity.com/t/illegal-instruction-when-importing-tensorflow/343894).
- Error: ```Error: ImportError: libcublas.so.9.0: cannot open shared object file: No such file or directory```.
- pip uninstall tensorflow
- pip install tensorflow-gpu
- pip uninstall tensorflow-gpu
- re-install nvidia driver
    ```bash
    sudo apt-get remove nvidia-*
    wget http://us.download.nvidia.com/XFree86/Linux-x86_64/375.66/NVIDIA-Linux-x86_64-375.66.run
    sudo bash ./NVIDIA-Linux-x86_64-375.66.run  --dkms
    ```
- followings seems worked.
    ```
    pip uninstall tensorflow-gpu
    pip install tensorflow-gpu==1.4
    ```
  but throws an error in the end.
    ```
    ResourceExhaustedError (see above for traceback): OOM when allocating tensor with shape[5,4096,5,18]
    ResourceExhaustedError (see above for traceback): OOM when allocating tensor with shape[32,64,160,576]
    ```

**Solution**: use AWS P2 instance type. 
- choose Udacity-carnd-advanced-deep-learning AMI
- choose P2 instance in GPU Compute instance type
- pip install tqdm

You are good to GO.


### Model coding
1. Load VGG. 
    - use ```tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)``` to load model.
    - get the default graph first ```tf.get_default_graph()```.
    - then, wee will use ```graph.get_tensor_by_name load``` to load followings from Pre-trained VGG model into the graph:
        - input
        - keep_prob
        - layer3_out
        - layer4_out
        - layer7_out
2. Build FNC Layers
    - Apply 1x1 Conv to layer7 output --> Upsample
    - Apply 1x1 Conv to layer4 output --> Skip Connection --> Upsample
    - Apply 1x1 Conv to layer3 output --> Skip Connection --> Upsample
    - Apply L2 Regularizer & Initializer to each of above steps
3. Define Cost & Optimization
4. Define training process


### Training
#### 1. With Only Regularizer
    ```
    EPOCH: 1  / 10  Loss: 1.870                                                                   
    EPOCH: 2  / 10  Loss: 0.726                                                                     
    EPOCH: 3  / 10  Loss: 0.653                                                                     
    EPOCH: 4  / 10  Loss: 0.623                                                                     
    EPOCH: 5  / 10  Loss: 0.577                                                                    
    EPOCH: 6  / 10  Loss: 0.595                                                                    
    EPOCH: 7  / 10  Loss: 0.572                                                                    
    EPOCH: 8  / 10  Loss: 0.510                                                                    
    EPOCH: 9  / 10  Loss: 0.554                                                                    
    EPOCH: 10  / 10  Loss: 0.512
    ```
- Loss did decearse.
- Let's see the results on randomly picked 3 images:
    ![alt text][round11]
    ![alt text][round12]
    ![alt text][round13]

**Result**: Good resart. But this is not what we want.    
    
#### 2. With Initializer & Regularizer
    ```
    EPOCH: 1  / 10  Loss: 0.517 
    EPOCH: 2  / 10  Loss: 0.372 
    EPOCH: 3  / 10  Loss: 0.253 
    EPOCH: 4  / 10  Loss: 0.324
    EPOCH: 5  / 10  Loss: 0.204
    EPOCH: 6  / 10  Loss: 0.185
    EPOCH: 7  / 10  Loss: 0.189
    EPOCH: 8  / 10  Loss: 0.190
    EPOCH: 9  / 10  Loss: 0.156
    EPOCH: 10  / 10  Loss: 0.105
    ```
- Loss looks much better.
- Let's see the results on randomly picked 3 images:
    ![alt text][round21]
    ![alt text][round22]
    ![alt text][round23]

**Result**: Much better. Looks GOOD.

#### 3. With only Initializer
    ```
    EPOCH: 1  / 10  Loss: 0.260
    EPOCH: 2  / 10  Loss: 0.346
    EPOCH: 3  / 10  Loss: 0.419
    EPOCH: 4  / 10  Loss: 0.273
    EPOCH: 5  / 10  Loss: 0.239
    EPOCH: 6  / 10  Loss: 0.117
    EPOCH: 7  / 10  Loss: 0.212
    EPOCH: 8  / 10  Loss: 0.374
    EPOCH: 9  / 10  Loss: 0.145
    EPOCH: 10  / 10  Loss: 0.115
    ```

  **Result**: Loss look good as last one. We can see thta Initializer is very important.

#### 4. Without Initializer & regularizer
    ```
    EPOCH: 1  / 10  Loss: 2.953
    EPOCH: 2  / 10  Loss: 1.172
    EPOCH: 3  / 10  Loss: 0.690
    EPOCH: 4  / 10  Loss: 0.658
    EPOCH: 5  / 10  Loss: 0.628
    EPOCH: 6  / 10  Loss: 0.636
    EPOCH: 7  / 10  Loss: 0.597
    EPOCH: 8  / 10  Loss: 0.599
    EPOCH: 9  / 10  Loss: 0.602
    EPOCH: 10  / 10  Loss: 0.602
    ```
  **Result**: Loss looks bad. We conclude that Initializer and Regularizer play important role in the FCN model training.

#### 5. EPOCHES = 20, Learning_rate = 0.0009
   ![alt text][lr0009]
   
  **Result**: Loss = 0.055, really good.


#### 6. EPOCHES = 20, Learning_rate = 0.0001
   ![alt text][lr0001]
   
  **Result**: Loss = 0.063, really good.

#### 7. EPOCHES = 20, Learning_rate = 0.00001
   ![alt text][lr00001]
   
  **Result**: Loss = 0.16, No good as the last two.
  
#### FINAL MODEL
- Hyper-parameters
    ```
    EPOCHS = 20 
    BATCH_SIZE = 32
    KEEP_PROB = 0.5
    LEARNING_RATE = 0.0009
    REG_SCALE = 1e-3   # L2 regularizer scale
    INI_STDDEV = 1e-3  # Initializer stddev
    ```
- Results
    ![alt text][round31]
    ![alt text][round32]
    ![alt text][round33]


## Resources
- [Udacity CarND Semantic Segmentation](https://github.com/udacity/CarND-Semantic-Segmentation).
- [FCN-8 Paper by Berkeley](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf).
- Helpful animations of convolutional operations, including transposed convolutions, can be found [here](https://github.com/vdumoulin/conv_arithmetic).

  

