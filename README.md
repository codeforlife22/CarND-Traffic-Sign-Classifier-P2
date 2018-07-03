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
[image9]: ./test_images/example_0006.png "Traffic Sign 6"
[image10]: ./examples/random_plot.png "random plot"

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

I randomly selected 4 images from the trannin set and plot them in the following figure, 
![alt text][image10]

As can been seen, they have different lighting and contrast to name a few. Diversity in the trainning set should help train a comprehensive model rather than focusing on a paticular set of images. 
### Design and Test a Model Architecture

#### 1. Preprocess data
The following techinques are used to preprocess data: 
1. Convert the images to grayscale because most of the traffic signs are color-independent. 
2. Normalization so that data has mean zero and equal variance

Here is an example of a traffic sign image before and after grayscaling.


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

Using the same learning rate in Lenet for number classification, 0.001 is a good default value. 

Technically, the large the EPOCH, the better the model. However, it takes longer time to train. Here 200 seems to be a reasonable number of iteraions and early termination is used to prevent over-trainning.  

As for the batch size, the large the batch size, the faster the model trains. But the memory on GPU/CPU is limited, so number 512 is choosen to tradeoff between speed and memory usage. 

Adaptive methods such as Adam optimizer is a little more complicated than stochastic gradient descent, and most of the time yields better performance. It is adopted here to train the model.

The model is trained on a local GPU (Nvidia Gforce 1060 with 4GB memory).

#### 4. Model performance

My final model results were:
* validation set accuracy of 95.1%
* test set accuracy of 93%

LetNet is adopted in this project as the architecture foundation, it has two CNN layer and three fully connected layer, Which seems to be sufficient for this project. However, with the default settings (as in the lecture), I was not able to reach the goal of validation accuracy (the highest I got was around 86%, which is far below expectation).

The reason is that 6 filters is not sufficient to extract enough features to accurately predict 43 different classes (as opposed to only 10 classes in the lecture). After increasing the number of filters to 64 and reduce the kernel size to 3x3, significant improvements were observed.  

### Test a Model on New Images

#### 1. Image set

Here are six German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9]

Among the new images, image 1 (roundabout), and 5 (20 km/h) should be a little harder to predict than the other images. For exmaple, image 2,3 4, should be easy to predict, as there is only one line(arrow) on a plain background in each image.    

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

The top_k logits for each image are listed below

Image 1 

| Scores         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 5059        			| Roundabout mandatory  									| 
| 3500    				| Priority road 										|
| 1675					| Right-of-way at the next intersection											|
| 1016	      			| 100 km/h					 				|
| 828				    | Slippery road   							|

Image 2

| Scores         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 6107        			| Keep right   									| 
| 2506    				| Turn left ahead									|
| 2219					| Slippery road											|
| 1191	      			| Road work			 				|
| 1181				    | Roundabout mandatory      							|

Image 3

| Scores         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 3749        			|   No entry 									| 
| 3306    				|  			Stop						|
| 2016					| Go straight or left					|
| 1979	      			| Turn right ahead				 				|
| 1856				    |  Keep left     							|

Image 4

| Scores         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 8066        			| Keep left 									| 
| 4153    				| Road work 										|
| 3130					| Go straight or left											|
| 2105	      			| Traffic signs				 				|
| 1452				    | Stop  							|

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

Image 6

| Scores         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 5453       			| Turn left ahead   									| 
| 3009    				| Ahead only										|
| 2018					| 30 km /h										|
| 1782	      			| Stop				 				|
| 1589				    | Go straight or left  							|


