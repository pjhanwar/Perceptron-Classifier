# Perceptron-Classifier
A vanilla and averaged Perceptron Classifiers for categorizing Hotel Reviews as True/Fake and Positive/Negative.

## Overview
Perceptron classifiers (vanilla and averaged) to identify hotel reviews as either true or fake, and either positive or negative using the word tokens as features from the text. 

## Data
A set of training and development data is uploaded contatining the following files:

1) train-labeled.txt containing labeled training data with a single training instance (hotel review) per line (total 960 lines). The first 3 tokens in each line are:
a) unique 7-character alphanumeric identifier
b) label True or Fake
c) label Pos or Neg
These are followed by the text of the review.

2) dev-text.txt with unlabeled development data, containing just the unique identifier followed by the text of the review (total 320 lines).

3) dev-key.txt with the corresponding labels for the development data, to serve as an answer key.

## Programs
It contains two programs: perceplearn.py to learn perceptron models (vanilla and averaged) from the training data, and percepclassify.py which uses the models to classify new data.

The learning program will be invoked in the following way:

> python perceplearn.py /path/to/input

The argument is a single file containing the training data; the program learns perceptron models, and writes the model parameters to two files: vanillamodel.txt for the vanilla perceptron, and averagedmodel.txt for the averaged perceptron. 

The classification program will be invoked in the following way:

> python percepclassify.py /path/to/model /path/to/input

The first argument is the path to the model file (vanillamodel.txt or averagedmodel.txt), and the second argument is the path to a file containing the test data file. 
The program reads the parameters of a perceptron model from the model file, classify each entry in the test data, and write the results to a text file called percepoutput.txt.

## Results on test data

Vanilla model:<br>

Neg 0.94 0.94 0.94<br>
True 0.84 0.82 0.83<br>
Pos 0.94 0.94 0.94<br>
Fake 0.83 0.84 0.83<br>
Mean F1: 0.8859<br>

{<br>
'Neg': {'fp': 9, 'fn': 10, 'tp': 150}, <br>
'True': {'fp': 26, 'fn': 28, 'tp': 132}, <br>
'Pos': {'fp': 10, 'fn': 9, 'tp': 151}, <br>
'Fake': {'fp': 28, 'fn': 26, 'tp': 134}<br>
}<br>

Averaged model:

Neg 0.91 0.93 0.92<br>
True 0.84 0.88 0.86<br>
Pos 0.93 0.91 0.92<br>
Fake 0.87 0.84 0.85<br>
Mean F1: 0.8890<br>

{
'Neg': {'fp': 14, 'fn': 11, 'tp': 149}, <br>
'True': {'fp': 26, 'fn': 20, 'tp': 140}, <br>
'Pos': {'fp': 11, 'fn': 14, 'tp': 146}, <br>
'Fake': {'fp': 20, 'fn': 26, 'tp': 134}<br>
}<br>

## Accuracy
The vanilla percpetron is 88.59% accurate in classifying test data calculated using above F1 measure.<br>
The averaged percpetron is 88.90% accurate in classifying test data calculated using above F1 measure.

