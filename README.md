# ML Graphics Predictor

This project combines Machine Learning (ML) techniques with Computer Graphics.
A Convolutional Neural Network (CNN) is used to predict the position and size of an object in 3D space, using only a 2D image of the space as input.
Computer graphics techniques are used to render the environment which generates the inputs for the CNN.

The model predicts four values:
- x: The horizontal position of the sphere. Ranges from -200 (left) to 200 (right).
- y: The vertical position of the sphere. Ranges from 100 (low) to 500 (high).
- z: The depth of the sphere. Ranges from -200 (far) to 200 (near).
- r: The radius of the sphere. Ranges from 25 to 75.

Run cnn.py to show 10 examples of images being predicted by a trained model.
