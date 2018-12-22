from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense


class MnistCnnModel(object):
    def __init__(self,
                 num_classes: int,
                 image_shape: tuple,
                 channels: int,
                 model: str = None) -> None:
        self.num_classes = num_classes
        self.image_shape = image_shape
        self.channels = channels
        self.model_path = model

    def get_model(self):
        input_shape = self.image_shape + (self.channels,)
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))
        return model

    def get_trained_model(self):
        model = self.get_model()
        model.load_weights(self.model_path)
        return model