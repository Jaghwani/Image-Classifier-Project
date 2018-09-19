### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

To run the code here you need to install following packages (torch, numpy, torchvision, collections, PIL, matplotlib, argparse, and json). The code should run with no issues using Python versions 3.*.
You need to download the images datasets here http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html , but I have added few samples of images for explanation purpose.
If you want to training and predicting using GPU you have to install Cuda toolkit when you are using Nvidia gpu's.

## Project Motivation<a name="motivation"></a>

This project is implemention of image classifier with PyTorch, the image classifier is to recognize different species of flowers.
I have trained this model using deep learning on thousands of images, this project is broken down into multiple steps:
1. Load and preprocess the image dataset
2. Train the image classifier on your dataset
3. Use the trained classifier to predict image content

## File Descriptions <a name="files"></a>

1. Image_Classifier_Project.ipynb : Here a notebook that will take you in real image classifer implementaion steps.
2. predict.py This uses a trained network to predict the class for an input image.
3. train.py : This will train a new network on a dataset and save the model as a checkpoint.
4. cat_to_name.json : This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.
5. flowers folder : is contain list of flowers images for training and testing purpose.
6. assets folder : is contain some images that used in Image_Classifier_Project.ipynb the notebook.


## Licensing, Authors, Acknowledgements<a name="licensing"></a>
Must give credit to Udacity courses for build this challange. Free to use the code here as you would like!
