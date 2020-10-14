# Comp137 DNN: Assiginment 3 

In this assignment, you will implement layers related to CNNs. Then you will implement a ResNet with `tensorflow.keras.layers` and apply it to an image classification problem. 

## Objective 

This assignment aims to help you understand CNNs. It also provide you a chance to tune a CNN -- the ResNet. 

## Instructions 

Run the notebook `notebook.ipynb` and follow instructions inside. 

To support the notebook, you will need to implement all functions in `np_layers.py`. Then you need to construct a ResNet in `conv_net.py`. After you have a ResNet constructed, you should tune the ResNet and find the best configuration. Then you should run the ResNet with your configuration in the notebook. 

You are strongly encouraged to use a [model tuner](https://www.tensorflow.org/tutorials/keras/keras_tuner) to search good hyperparameters. 

## Grading

Detailed grading rules are inside the notebook. Finally, you need to submit 
* all files in this folder;
* a folder containing your trained model. 

(We may decide not to submit the model if gradescope's size limit does not allow so).  

To submit these files in a zip file, please zip all files in the folder `assignment3` into a zip file. All files/folders in `assignment3` should be at the root of the zip file. Note please do NOT zip the assignment3 folder. If you use linux, you can run`zip -r assignment2.zip *` WITHIN the `assignment3` folder. 

Your implementations of different layers and your trained model will be graded by the autograder through GradeScope.

When we grade your code, we check your result notebook as well as your code. If your code cannot generate the result of a problem in your notebook file, you will get zero point for that problem. Actually, if you follow instructions in the notebook and finish all tasks, you are likely to get most points. 


