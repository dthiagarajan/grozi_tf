## Using TensorFlow for CNN's for recognizing *in situ* groceries

Welcome!

This repository houses all work towards using TensorFlow and other CV tools to recognize *in situ* groceries at a particular store (Mattin's, the store located in Duffield Hall at Cornell). All work here was done during the summer of 2016 under the instruction of Dr. Serge Belongie.

### Table of Contents

[Notes](https://github.com/dthiagarajan/grozi_tf#notes)

[Current Tasks](https://github.com/dthiagarajan/grozi_tf#current-tasks)

[Completed Tasks](https://github.com/dthiagarajan/grozi_tf#completed-tasks)

[Project Description](https://github.com/dthiagarajan/grozi_tf#project-description)

1. [Introduction](https://github.com/dthiagarajan/grozi_tf#introduction)
2. [Data and Methodology](https://github.com/dthiagarajan/grozi_tf#data-and-methdology)
3. [References](https://github.com/dthiagarajan/grozi_tf#references)

[Archives](https://github.com/dthiagarajan/grozi_tf#archives)

### Notes:
These are all in reverse chronological order to keep track of recent updates more easily. (6/13/16) I'll be removing updates that aren't as relevant to current tasks, so everything from 6/1/16 and before will be at the bottom of this README.

####6/15/16
Increasing the size of the training set improves the accuracy of the CIFAR-10 network to around 96% consistently, still with distorted input. The following is the ROC curve for various thresholds:
![ROC Curve for Tide images trained on CIFAR-10 network](/tide/ROC_tide_cifar10network.png)

Clearly, it's not performing that well, as the uppermost, leftmost point is not very close to (0,1), so we're trying to see how well the Alexnet network does with this dataset. However, we're running into some trouble with the rank of the output. Specifically, when calculating the loss, we get the following error: regarding the logits:
```
ValueError: Shape (128, 6, 6, 256) must have rank 2
```

####6/13/16
Using the CIFAR-10 code, we've augmented the training and testing data set to work with distorted inputs as well (shearing, translation, scaling, etc.), and the precision after running about 8000 epochs was about 92%, so we will run several more epochs to see if that improves the accuracy. If that doesn't work, we'll try modifying the network structure to make it more similar to the structures used in AlexNet and ImageNet and see if that improves the accuracy. If that doesn't work, we'll research more closely to find a better suited network structure.

####6/9/16
We've now altered the CIFAR-10 code provided from TensorFlow's tutorial pages that uses a CNN for recognition to now distinguish the presence of Tide in a 32 x 32 image. At the moment, it gets everything right when testing (i.e. precision of 1) with the given network structure (the same one used for the CIFAR-10 dataset). Now, we have to figure out how to graph the ROC curve - this may require messing around with the following line of code in cifar10_eval.py:
```
top_k_op = tf.nn.in_top_k(logits, labels, 1)
```
Specifically, we need to vary the threshold that determines if a logit matches with a label, and plot the according pairs of false positive rates and true positive rates (or 1 - specificity versus the sensitivity). From here, if the ROC curve looks more or less like what was obtained on distinguishing just Tide in the previous study, we will move on to more difficult products, in addition to incorporating more classes for classification.

To actually alter the code, we will inspect the logits tensor and the labels tensor, and figure out how the method works to determine the relevant data for plotting the ROC curve.

####6/7/16
Now working on using TF to build a network that recognizes Tide on the shelf, as mentioned in the second current task. To do so, I'm essentially using the code available from TensorFlow's repo using CNN's with the CIFAR-10 dataset, and modifying it to work on images that either do or do not have Tide. To do so, the images in my dataset need to be reshaped to be 32 x 32. Then, they need to each be flattened, and the associated label byte should be added to the front of the flattened image information, as mentioned on this [page](http://stackoverflow.com/questions/35032675/how-to-create-dataset-similar-to-cifar-10/35034287).

Another idea I had was to use a higher-level framework built with TensorFlow, such as [TFLearn](http://tflearn.org/). I think this might make it much easier to actually code the various neural network structures to try different networks for training on various images, but as we go on, I'll keep looking for similar high-level frameworks. This will accordingly be added to the related work section of the proposal, but we will still keep options open for using TensorFlow solely.



### Current Tasks
1. Use distorted input methods provided by TF to increase input size and test on the distorted inputs as well, and plot the associated ROC curve to evaluate the network. Based on the results, modify the network structure using precedents like AlexNet or ImageNet to try and get better accuracies if necessary.
2. Pick 9 more products that aren't as easily distinguished, and use TF to do recognize the 10 products (including Tide).
2. Collect data from Mattin's for actual training.

### Completed Tasks
1. Become familiar with relevant computer vision tools, find other relevant application studies, and build up related work section of proposal based on this.
2. Pick one product from GroZi-120 (easiest choice would be Tide), and use TF (training and testing) to detect it in 20 shelf images, where half contain Tide, and half don't.

### Project Description
#### Introduction
As the need for assistive technology for the visually impaired becomes more prominent and feasible with the use of machine learning, we revisit the problem of using pictures of objects taken in ideal conditions to recognize more common scenes of these same objects. This problem encompasses several possible applications, but in this study, we will specifically look at grocery products sold in Mattin's. In essence, we will revisit the study conducted by Merler, Galleguillos, and Belongie, but with a more modern approach involving the tools contained in TensorFlow. 

The main problem we hope to address by conducting this study is paramount to maintaining the state of assistive technology: ideally, training data and testing data would be obtained from the same distribution of data. However, in real-world application, it is more convenient to obtain training data from more well-kept databases, such as the web or dedicated databases. As a result, using the tools offered in TensorFlow, we intend to design a portable system that can recognize the groceries in Mattin's, trained on images taken from both an ideal and realistic environment.

Specifically, the purpose of this study will be to build a database of images encompassing the inventory of Mattin's ranging from ideal to realistic shots, as well as to use various approaches to actually recognize the grocery products in Mattin's. This will include color histogram matching, SIFT matching, compared with training a neural network that takes image pairs as input (where the image pair consists of an ideal and realistic shot of a grocery object in Mattin's) for classification.
#### Data and Methdology
To begin, we will need to obtain the relevant types of data for input. As mentioned in the previous study, one database that can be used is the (http://grozi.calit2.net/grozi.html - GroZi-120 database), which is a database of 120 products with images of objects, ranging over various attributes of the image, and where each product has two different representations: either \textit{in situ}, i.e. in a realistic environment, or \textit{in vitro}, i.e. in an idealistic environment. Another possible way to scrape data could be to use Google searches, and take a sample of the top image hits as training/test data.

Additionally, data would be obtained from Mattin's in person, and due to the changing inventory, images and videos can be taken periodically of each product in the whole inventory to better recognize the inventory, given the varying nature of the products sold. To better train the network being used, photos and videos taken in person will vary in scale and lighting to simulate a more realistic environment.

Once the relevant data has been obtained, it will be used independently on each localization and recognition algorithm, as well as arranged into pairwise input to be used for training on a convolutional neural network built in TensorFlow. Specifically, we will build a sample of several neural networks built in TensorFlow (which will be chosen from the set of all possible networks by using distinctive heuristics) and see how they perform compared to each other on various samplings of training and testing data. Once we have selected the network that performs best, we will compare the results of the standard algorithms with the new accumulation of data to the results in the previous study, as well as comparing how the chosen network built in TensorFlow performs compared to other algorithms on the new accumulation of data.
#### References
Merler, Michelle; Galleguillos, Carolina; Belongie, Serge. *Recognizing Groceries in situ Using in vitro Training Data*. SLAM, Minneapolis, MN, 2007.

### Archives

####6/1/16
Update to the error from the very beginning - it turns out that the installation page on [Tensorflow](https://www.tensorflow.org/versions/0.6.0/get_started/os_setup.html#pip_install) has an outdated version. I installed the 0.8.0 version of TensorFlow, and now all dependencies are there, and the file that works with MNIST data works as detailed in the tutorials.

####5/29/16
Figured out why the error kept coming up: the input_data Python file, and the associated imports necessary, are all in a different directory in the TensorFlow soure code (i.e. from their Github), and was not installed through pip. I tried working around that by cloning their repo locally, but that won't work unless I move the relevant files into the folder of my local installation of TensorFlow, which brings up the problem of duplicate folder names. 

As a result, if I were to change those, I'd have to go through almost all of their source code and fix up the import statements based on my own changes, so I'm choosing to just skip this for now, at least until it proves to be an impasse. At the moment, everything else works (in the scope of everything that I've tried/ran so far), so I'll just continue working with the framework, and hopefully, this won't pose a problem over the course of the summer.

####5/25/16
The tf_tutorial.py file is not working: when run, the following error occurs:
```
Traceback (most recent call last):
  File "tf_tutorial.py", line 1, in <module>
    import tensorflow.examples.tutorials.mnist.input_data
ImportError: No module named examples.tutorials.mnist.input_data
```
I tried running the input_data.py file to work around that, but that doesn't work as well, giving the following error:
```
Traceback (most recent call last):
  File "input_data.py", line 29, in <module>
    from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
ImportError: No module named contrib.learn.python.learn.datasets.mnist
```
Currently, I'm working on fixing this.