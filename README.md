# self-driving-car-semantic-segmentation

[//]: # (Image References)
[train_label1]: ./assets/train_label1.PNG
[train_label2]: ./assets/train_label2.PNG


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

## Training & Testing

### AWS Setup

### Model coding

### Training

## Resources
- [Udacity CarND Semantic Segmentation](https://github.com/udacity/CarND-Semantic-Segmentation).
- [FCN-8 Paper by Berkeley](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf).

  

