# EmojiRecognition
Raw implementation of MNIST cnn trained for 2389 classes of apple emojis

Based on [MNIST CNN](https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py), using [emoji database](https://github.com/SHITianhao/emoji-dataset) as basis
 of my augmented one (need to pull that).
 
 All preparation functions are placed in ```augment.py```, '''train.py''' describes model and how to interact with data.
 Also you can just use ```final.h5``` and ```predict.py``` to predict emojis.
 
 # NOTICE:
The model was trained on .png file with 4 channels but this .pngs were read as 'rgb' so in ```predict.py``` you can see strange behaviour of adding green background to recognizable .png.

# TODO:
Train again with greyscale
