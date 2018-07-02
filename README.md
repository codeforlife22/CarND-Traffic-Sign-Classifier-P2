# **Traffic Sign Recognition** 


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/histogram.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./test_images/example_00001.png "Traffic Sign 1"
[image5]: ./test_images/example_00002.png "Traffic Sign 2"
[image6]: ./test_images/example_00003.png "Traffic Sign 3"
[image7]: ./test_images/example_0004.png "Traffic Sign 4"
[image8]: ./test_images/example_0005.png "Traffic Sign 5" 
[image9]: ./test_images/example_0006.png "Traffic Sign 5"

### Data Set Summary & Exploration

#### 1. Summary of the data set. 

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 12630
* The size of test set is 4410 
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Visualization of the dataset.

Here is an exploratory visualization of the data set (a histogram of the tranning set).

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Preprocess data
The following techinques are used to preprocess data: 
1. Convert the images to grayscale because most of the traffic signs are color-independent. 
2. Normalization so that data has mean zero and equal variance

Here is an example of a traffic sign image before and after grayscaling.

Addition

![alt text][image2]


#### 2. Model architecture
The model is based on LeNet for number classification, but some changes are included. 
The patch size is changed to from 5x5 to 3x3 and the the number of filters is increased to 64 (from 6).
Based on empirical observations, increasing the number of filters significantly improves the accuracy of the model. 

It consists of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input (L1)        | 32x32x1 Grayscale image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 30x30x64 	|
| RELU					|												|
| Dropout					|	With dropout probability = 50%											|
| Max pooling	      	| 2x2 stride, valid padding, outputs 15x15x64 				|
| Convolution 5x5 (L2)  | 1x1 stride, valid padding, outputs 11x11x64 	|
| RELU					|												|
| Dropout					|	With dropout probability = 50%											|
| Max pooling	      	| 2x2 stride, valid padding, outputs 5x5x64 				|
| Flatten | input 5x5x64, output 1600 |
| Fully connected	(L3)	| input 1600, output 120  									|
| RELU					|												|
| Fully connected	(L4)	| input 120, output  84 						|
| RELU					|												|
| Fully connected	(L5)	| input 84, output  43 |
| One hot key encoding |        									|
| Softmax				| softmax with cross entropy        									|
|	Optimizer					|							AdamOptimizer					|

#### 3. Trainning the model
The following paramers are used to train the above model:
1. learning rate = 0.001
2. EPOCH = 200  (with early termination when validation accuracy >= 95%)
3. batch size = 512

The model is trained on a local GPU (Nvidia Gforce 1060 with 4GB memory)

#### 4. Model performance

My final model results were:
* validation set accuracy of 95.1%
* test set accuracy of 93%
 

### Test a Model on New Images

#### 1. Image set

Here are six German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9]


#### 2. Prediction
The model predicts correctly 6 out of 6 images, which yields an accuracy of 100%, which compares favorably with the test set accuracy of 93%.
Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Roundabout mandatory      		| Roundabout mandatory   									| 
| Keep right     			| Keep right 										|
| No entry					| No entry											|
| Keep left     			| Keep left 										|
| 20 km/h	      		| 20 km/h					 				|
| Turn left ahead			| Turn left ahead      							|


#### 3. Confidence level

For the given new images, the model is relatively sure of its predictions, with the top prediction close to 1.0 for all the cases.
As for each image, I select to output the top 5 logits (before softmax function) to get a better understanding of which top five prediction scores (as probabilities other than the top one will be so close to zero, top_k is not able to select sensibly)
Here is an example for the 5th image:

| Scores         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 7689        			| 20 km/h   									| 
| 5572    				| 30 km /h 										|
| 2702					| 120 km/h											|
| 2518	      			| 80 km/h					 				|
| 2089				    | 70 km/h      							|

As can be seen from the above table, these are very close numbers which sometime hard to distinguish, even from human eyes. 
Overall, this model performs satisfactorily. 



