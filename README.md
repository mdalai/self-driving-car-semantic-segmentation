# self-driving-car-semantic-segmentation

[//]: # (Image References)
[train_label1]: ./assets/train_label1.PNG
[train_label2]: ./assets/train_label2.PNG
[FCN_arch]: ./assets/FNC_architecture.PNG


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

### Training

## Resources
- [Udacity CarND Semantic Segmentation](https://github.com/udacity/CarND-Semantic-Segmentation).
- [FCN-8 Paper by Berkeley](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf).
- Helpful animations of convolutional operations, including transposed convolutions, can be found [here](https://github.com/vdumoulin/conv_arithmetic).

  

