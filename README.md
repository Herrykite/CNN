Conv Net
===========================
Author : WangHairui

# Requirements
Python 3.8.3\
Cuda compilation tools, release 11.0, V11.0.221\
Pytorch 1.6.0\
sys\
yacs\
pickle\
yaml\
json\
os\
torchvision


# How To Use
1. Execute the compiling file tools/preprocess_data.py, which will generate a CNN_output_data folder for you in the output folder and store the preprocessed data file in the format.data.
2. Execute the compiling file engine/trainer.py, and you can start training or testing (if you need to test, just annotate the train(epoch) call in the run())
3. After normal execution, you will find that the output folder contains CNN_output_parameter, CNN_test_output, CNN_output_data, and log.txt. The CNN_output_parameter is used to hold the network and optimizer parameters, requiring 2.56GB of storage per save. CNN_test_output is used to output the.obj file of the test image.
4. If you need to continue with the last argument, uncomment the following sections in the run()：
    ~~~~ 
     net.load_state_dict(torch.load(cfg.OUTPUT.PARAMETER + cfg.OUTPUT.SAVE_NET_FILENAME)) \
     print('loaded net successfully!')
     optimizer.load_state_dict(torch.load(cfg.OUTPUT.PARAMETER + cfg.OUTPUT.SAVE_OPTIMIZER_FILENAME)) \
     print('loaded optimizer successfully!')
At the same time, you will need to modify the load model network parameters and optimizer parameters within defaults.py. \
This is explained in detail in the defaults.py file.



# Introduction
**defaults.py** \
This is used to store file paths and model parameters \
**trainer.py** \
This is used for training and testing \
**cnn.py**  \
This includes the convolutional neural network, and Kaiming Initialization method, consisting of 12 convolutional layers and 6 activation functions \
**deal_with_obj.py** \
This is used to read and write .obj files \
**draw.py** \
This is used to visualize the training process \
**preprocess_data.py** \
This is used to preprocess the training label data \
**datasets_transform.py**  \
This is used to process input image data, including normalization, data validation, and data enhancement \



# Problem Specification
1. At present, there is a problem that there is not enough video card memory for training after loading training parameters and optimizing parameters, but it can be predicted if only testing is carried out without training.
