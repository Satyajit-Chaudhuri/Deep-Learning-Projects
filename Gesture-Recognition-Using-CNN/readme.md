## Project Name - Gesture Recognition using CNN & RNN architectures.

## Premise : 
- Gesture recognition can be seen as a way for computers to begin to understand human body language, thus building a better bridge between machines and humans than older text user interfaces or even GUIs (graphical user interfaces), which still limit the majority of input to keyboard and mouse and interact naturally without any mechanical devices. The following research work focuses on devising a neural network architecture that can recognise five different gestures performed by the user. This architecture is then to be employed by a home electronics company which manufactures state of the art smart televisions to help users control the TV without using a remote.


## Objective: 

The following research work deals with recognising five different gestures performed by the user which will help users control the TV without using a remote.

The five gestures are:

- Thumbs up: Increase the volume.
- Thumbs down: Decrease the volume.
- Left swipe: 'Jump' backwards 10 seconds.
- Right swipe: 'Jump' forward 10 seconds.
- Stop: Pause the movie.

The Key objective of the study is to build a neural network architectures and train them on the 'train' folder to predict the action performed in each sequence or video and which performs well on the 'val' folder as well.



## Architecture Used:

- Convulation 3D

- Convulation 2D

- LSTM

- GRU

- RESNET

- MOBILENET


## Methodology:
For analysing videos using neural networks, two types of architectures are used commonly. One is the standard CNN + RNN architecture in which the images of a video are passed through a CNN which extracts a feature vector for each image, and then the sequence of these feature vectors are passed through an RNN. The other popular architecture used to process videos is a natural extension of CNNs - a 3D convolutional network. A 3D CNN uses a three-dimensional filter to perform convolutions. The kernel is able to slide in three directions, whereas in a 2D CNN it can slide in two dimensions. Based on the very same tenets, seven distinct experimental models have been developed, each with a different parameter or architectural setting.Here we make the model using different functionalities that Keras provides. We use `Conv3D` and `MaxPooling3D` and not `Conv2D` and `Maxpooling2D` for a 3D convolution model. We also use `TimeDistributed` while building a Conv2D + RNN model. The last layer is the softmax. The models have then been trained on the training data and validated consequently. The architecture with the best validation accuracy has finally been selected for evaluation on the test dataset.

## Model - 1

### Parameters:
- Batch_Size = 8, 
- Image_Size = (180,180), 
- Epochs = 16

### Intuition:

- The first model is the base model on which the later models have been built. This model has seen a batch size of 8. The image dimensions are (180,180) and the model is trained on 16 epochs. The kernel size has been kept as (3,3,3) and drop outs have been used after 3D Convolution layers and dense layers. The model uses four convolution layers. The first layer has a filter size of 32, followed by 64, which is again followed by a convolution layer of filter size 64 and 32 for the last layer. The architecture then saw a dense layer of 512 neurons followed by the output layer with 5 neurons.



## Model - 2

### Parameters:
- Batch_Size = 20, 
- Image_Size = (160,160), 
- Epochs = 16

### Intuition:
- The batch size is increased to 20 and the image size is reconfigured to (160,160). The architecture used eight convolutions with kernel size of (3, 3, 3). This is then followed by two dense neural networks. The first convolution network has 16 filters, the second and third network has 32 filters, the fourth, fifth and sixth convolution layers have 64 filters and the last two convolution layers have 128 filters. The activation functions used in the convolution nets is “relu”. This is then followed by two dense networks of 64 neurons. The dropouts are being introduced after each dense layer. The final dense layer has 5 output neurons with softmax.


## Model - 3

### Parameters:
- Batch_Size = 12, 
- Image_Size = (160,160), 
- Epochs = 16

### Intuition:

As increasing the number of convulations layers did not result into increase in validation accuracy and hence in this trial the number of layers have been reduced keeping the batch size and the image size same. The kernel size and the activation functions have also been kept same. The Batch Normalization has been applied after each convulation layer. The max pooling layers of pool size (2, 2, 2) have been used after the second, third and fourth convulation layers. The number of epochs is maintained at 16. The model is composed of four convulation layers. The first and second layer has 32 filters, the next has 64 filters, the last layer has 128 filters. The dense networks and the dropouts have been kept the same as in the previous model.

## Model - 4

### Parameters:
- Batch_Size = 12, 
- Image_Size = (160,160), 
- Epochs = 16

### Intuition:
- In this model, the architecture uses the Long Short Term Memory Cells. The model uses four two-dimensional convulation layers having 16 filters and kernel size of (2,2). The number of epochs have been kept at 16 like previous models only. The activation functions employed here is rectified linear unit. Batch Normalization has been used after each convulation layer. The max pooling layers have been used after the first two layers and a dropout layer of 0.2 dropout. The next max pooling and dropouts have been employed after the third and fourth layer. 
All these layers are time distributed networks in line with the RNN tenets. After the convulation layers, the LSTM cell with 256 neurons cells have been employed with a dropout of 0.5. This is followed by a dense neuron network of 64 neurons and dropout of 0.25. The last layer is the output with 5 output neurons and softmax activation

## Model - 5

### Parameters:
- Batch_Size = 12, 
- Image_Size = (160,160), 
- Epochs = 16

### Intuition:
- In this model we use the concept of Gated Recurrent Units. The key difference between GRU and LSTM is that GRU's bag has two gates that are reset and update while LSTM has three gates that are input, output, forget. GRU is less complex than LSTM because it has less number of gates. If the dataset is small then GRU is preferred otherwise LSTM for the larger dataset. Thus the GRU model is being used in this study to see if it is more efficient for this particular task. In this model, the batch size has been kept at 12 , image sizes (160.160) and number of epochs at 16. The model is built up of 4 convulational units. The first layer has 16 filters, followed by 32 filters. This is being followed up by 64 filters and last CNN layer has 128 filters. The kernel size is kept at (2,2) for first layer and (3,3) for remaining layers. The batch normalization and max pooling has been performed after each CNN layer.This is then folllowed up by the GRU layer of 256 neurons. This is then succedded by a dropout layer. The remaining network is composed of densely connected neural nets finally giving an output of 5 classes with softmax activation.

## Model - 6

### Parameters:
- Batch_Size = 12, 
- Image_Size = (160,160), 
- Epochs = 16
- Transfer Learning Model :  RESNET

### Intuition:
- The study now focusses on using the Resnet architecture to convolve on the images which is then coupled with the Gated Recurrent Units to make predictions on the validation set. Here also, the batch sizes are kept at 12m, image sizes at (160,160) and epoch is set to 12. The model uses the weights similar to that of the resnet model. After the CNN layer a Batch Normalization layer and max pooling is applied. This is then followed by GRU units of 128 neurons. This is then followed by dense network of 64 neurons and the final output neuron has 5 output neurons with softmax function.



## Model - 7

### Parameters:
- Batch_Size = 12, 
- Image_Size = (160,160), 
- Epochs = 16
- Transfer Learning Model :  MOBILENET

### Intuition:

- The transfer learning model Mobilenet is now being used to train the model. MobileNet is a streamlined architecture that uses depthwise separable convolutions to construct lightweight deep convolutional neural networks and provides an efficient model for embedded vision applications. Now this advantage of mobilenet is being harnessed to work alongwith the Gated Recurrent Units to test on the validation data. Here also, the batch sizes are kept at 12m, image sizes at (160,160) and epoch is set to 12. The model uses the weights similar to that of the resnet model. After the CNN layer a Batch Normalization layer and max pooling is applied. This is then followed by GRU units of 128 neurons. This is then followed by dense network of 64 neurons and the final output neuron has 5 output neurons with softmax function.



## Results:



| Model Number | Batch Size | Image Size | Epochs | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 1 | 8 | (180,180) | 16 | 0.174 | 0.951 | 0.8 | 0.820 |
| 2 | 20 | (160,160) | 16 | 0.30746 | 0.904 | 3.53 | 0.210 |
| 3 | 12 | (160,160) | 16 | 0.121 | 0.965 | 0.435 | 0.820 |
| 4 | 12 | (160,160) | 16 | 0.096 | 0.98 | 0.629 | 0.78 |
| 5 | 12 | (160,160) | 16 | 0.211 | 0.93 | 0.409 | 0.920 |
| 6 | 12 | (160,160) | 12 | 0.509 | 0.82 | 1.04 | 0.620 |
| 7 | 12 | (160,160) | 12 | 0.0176 | 0.996 | 0.151 | 0.96 |














## Acknowledgements
This project was inspired by IIITB & Upgrad.
