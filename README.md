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

#### 7/27/16
Added TFLearn implementation of fine-tuning on VGG-16 network (the only pre-trained model available for TFLearn). The TFLearn implementation is very memory intensive and I've been unable to even test it fully on my local machine. I'm still investigating why the Inception model didn't run into the same problems. Further work is needed to implement randomized backgrounds live during training, attempting to recreate the [Held, Thrun, Savarese (2015)](https://github.com/dthiagarajan/grozi_tf/blob/master/proposal#data-augmentation) study, and getting TensorFlow up and running on AWS.

#### 7/6/16
The results of fine-tuning the [Inception v3](http://arxiv.org/abs/1512.00567) network on our data set were underwhelming based on the ROC curves. Some classes which were sufficiently distinctive (e.g. 4 - Cheerios, 34 - Tide) performed well while those for which there was little data (e.g. 16 - Snyder's Pretzels) or pose varied too much (e.g. 59 - Dr. Pepper) performed very poorly:
![ROC Curve 4 vs Rest](/roc_curves/roc_curve_4_vs_rest.png)
![ROC Curve 34 vs Rest](/roc_curves/roc_curve_34_vs_rest.png)
![ROC Curve 16 vs Rest](/roc_curves/roc_curve_16_vs_rest.png)
![ROC Curve 59 vs Rest](/roc_curves/roc_curve_59_vs_rest.png)

Most, however, looked more like this:
![ROC Curve 22 vs Rest](/roc_curves/roc_curve_22_vs_rest.png)

These were generated using undistorted training and testing inputs. For the tide/not-tide binary case, I trained and tested with random cropping, scaling, and brightness changes and no random flipping. The ROC curves look like this:
![ROC Curve Tide vs Rest](/roc_curves/roc_curve_tide_vs_rest.png)
![ROC Curve Not-Tide vs Rest](/roc_curves/roc_curve_not_tide_vs_rest.png)

I would like to do a comparison of the ROC curves from the augmented data set vs. non-augmented data set to determine the impact with our data.

#### 6/28/16
While using the other networks do show some signs of improvement over CIFAR-10, overall, the improvement isn't significant enough, so as a short-term measure, we are going to try jittering the data and preprocessing a bit more closely to see if this brings about any improvement. More specifically, this is easily implemented using TFLearn's API by adding some parameters to the input layer:
```
# Real-time image preprocessing
img_prep = tflearn.ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = tflearn.ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=20.0)
```
In the following days, the networks will be run again to evaluate their performance on all 120 classes. From here, we hope to better implement are metrics of evaluation, as well as to start researching things we may do to tweak the actual architecture of the net that we use in the final stages.

### Current Tasks
1. Use distorted input methods provided by TF to increase input size and test on the distorted inputs as well.
2. Implement various metrics to more precisely assess the accuracy of our model, including plotting the associated ROC curve to evaluate the network. Based on the results, modify the network structure using precedents like AlexNet or ImageNet to try and get better accuracies if necessary.


### Completed Tasks
1. Become familiar with relevant computer vision tools, find other relevant application studies, and build up related work section of proposal based on this.
2. Pick one product from GroZi-120 (easiest choice would be Tide), and use TF (training and testing) to detect it in 20 shelf images, where half contain Tide, and half don't.

### Project Description

For more specific details regarding the motivation for this project, as well as milestones we hope to achieve, see our [proposal](https://github.com/dthiagarajan/grozi_tf/blob/master/proposal/Proposal.pdf).

### Archives

####6/20/16
Clearly, the ROC curve wasn't indicative of the performance we were hoping for, which makes the 96% precision recall somewhat redundant, so while it is necessary to improve that, we are going to continue to move forward and expand our network to handle all 120 classes of the GroZi dataset. From this, we hope to see improved performance simply from the higher probability of allowing for two things to be distinguished.

To actually get the ROC curves, we will do binary classification on each class. This somewhat contradicts the notion that we hope for the ROC to improve, and thus, we'll be testing out some other network structures on all 120 classes to see if there is any positive effect. So far, the Alexnet structure has been implemented (thanks to TFLearn), and in the near future, such as ResNet and VGG.

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
