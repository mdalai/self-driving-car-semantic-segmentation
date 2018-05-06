import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
#import csv


#######--HYPER PARAMETERS--############################
EPOCHS = 20 #10
BATCH_SIZE = 32
KEEP_PROB = 0.5
LEARNING_RATE = 0.0009 #0.00001 #0.0001 #0.0009
REG_SCALE = 1e-3   # L2 regularizer scale
INI_STDDEV = 1e-3  # Initializer stddev

NUM_CLASSES = 2
IMAGE_SHAPE = (160, 576)

# Work Directory Settings
DATA_DIR = './data'
RUNS_DIR = './runs'
#MODEL_SAVE_FILE = './model/FCN_train_model.ckpt'

# save loss in csv file
#LOSS_FILE = 'loss.csv'


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    # get the graph first
    graph = tf.get_default_graph()
    # load followings into the graph by names
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return image_input, keep_prob, layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify  - pixel is road OR not road
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    ####### Layer7 - 1x1 Convolution  #######################
    # Add L2 Regularizer to prevent overfitting to each layer
    # Add Initializer that generate tensors with a normal distribution to each layer
    layer7_conv_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, kernel_size=1, strides=(1,1), padding='same', kernel_regularizer = tf.contrib.layers.l2_regularizer(REG_SCALE), kernel_initializer=tf.random_normal_initializer(stddev=INI_STDDEV) )
    ####### Layer7 1x1 Conv output  - Upsample ##############
    output = tf.layers.conv2d_transpose(layer7_conv_1x1, num_classes, kernel_size=4, strides=(2,2), padding='same', kernel_regularizer = tf.contrib.layers.l2_regularizer(REG_SCALE), kernel_initializer=tf.random_normal_initializer(stddev=INI_STDDEV) )

    ####### Layer4 - 1x1 Convolution  #######################
    layer4_conv_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, kernel_size=1, strides=(1,1), padding='same', kernel_regularizer = tf.contrib.layers.l2_regularizer(REG_SCALE), kernel_initializer=tf.random_normal_initializer(stddev=INI_STDDEV) )
    ####### Skip connection  ################################
    input = tf.add(output, layer4_conv_1x1)
    ####### Layer4 above 2 process  - Upsample ##############
    output = tf.layers.conv2d_transpose(input, num_classes, kernel_size=4, strides = (2,2), padding='same', kernel_regularizer = tf.contrib.layers.l2_regularizer(REG_SCALE), kernel_initializer=tf.random_normal_initializer(stddev=INI_STDDEV) )

    ####### Layer3 - 1x1 Convolution  #######################
    layer3_conv_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, kernel_size=1, strides=(1,1), padding='same', kernel_regularizer = tf.contrib.layers.l2_regularizer(REG_SCALE), kernel_initializer=tf.random_normal_initializer(stddev=INI_STDDEV) )
    ####### Skip connection  ################################
    input = tf.add(output, layer3_conv_1x1)
   ####### Layer3 above 2 process  - Upsample ###############
    output = tf.layers.conv2d_transpose(input, num_classes, kernel_size=16, strides = (8,8), padding='same', kernel_regularizer = tf.contrib.layers.l2_regularizer(REG_SCALE), kernel_initializer=tf.random_normal_initializer(stddev=INI_STDDEV) )

    # print the dimension
    #tf.Print(output, [tf.shape(output)[1:3]])


    return output

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    ### Reshape 4D tensor to 2D tensor
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name="logits_2d")
    ### Reshape 4D tensor to 2D tensor
    labels = tf.reshape(correct_label, (-1, num_classes),name="labels_2d")
    ### use Cross Entropy Loss
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= logits, labels= labels), name = 'loss')
    ### adapt Adam optimizer
    train_op = tf.train.AdamOptimizer(learning_rate= learning_rate).minimize(cross_entropy_loss)    

    return logits, train_op, cross_entropy_loss


tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    losses =[]
    # TODO: Implement function
    for epoch in range(epochs):
        for images, labels in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss], 
                feed_dict = { input_image: images, correct_label: labels, keep_prob: KEEP_PROB, learning_rate: LEARNING_RATE })
                       
        print("EPOCH: {}".format(epoch + 1), " / {}".format(epochs), " Loss: {:.3f}".format(loss) )

        losses.append('{:3f}'.format(loss))
    
    print()
    print(losses) 
    # save the model
    #saver.save(sess, MODEL_SAVE_FILE) 
    
tests.test_train_nn(train_nn)


def run():

    tests.test_for_kitti_dataset(DATA_DIR)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(DATA_DIR)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(DATA_DIR, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(DATA_DIR, 'data_road/training'), IMAGE_SHAPE)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function

        ####### Load VGG ######################    
        image_input, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        ####### Build FCN Layers ##############
        nn_last_layer = layers(layer3_out, layer4_out, layer7_out, NUM_CLASSES)

        # define placehoder for labels
        labels = tf.placeholder(tf.int32, [None, None, None, NUM_CLASSES], name='labels')
        # define placehoder for learning_rate
        learning_rate = tf.placeholder(tf.float32, name ='learning_rate')

        ####### Cost & Optimization ###########
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, labels, learning_rate, NUM_CLASSES)

        # initialize variables      
        sess.run(tf.global_variables_initializer())

        # TF saver object
        saver = tf.train.Saver()

        # TODO: Train NN using the train_nn function
        train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, cross_entropy_loss, image_input, labels, keep_prob, learning_rate)

        # TF save the graph and checkpoint
        saver.save(sess,'data/models/model1',global_step=1000)

        # TODO: Save inference data using helper.save_inference_samples
        #helper.save_inference_samples(RUNS_DIR, DATA_DIR, sess, IMAGE_SHAPE, logits, keep_prob, image_input)

        # OPTIONAL: Apply the trained model to a video

if __name__ == '__main__':
    run()
