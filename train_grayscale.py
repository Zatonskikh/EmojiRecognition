
from __future__ import print_function

import keras
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras_preprocessing.image import ImageDataGenerator

import settings
from models import MnistCnnModel


# using this (https://stackoverflow.com/questions/43076609/how-to-calculate-precision-and-recall-in-keras/43104549) to compute precision and recall
def as_keras_metric(method):
    import functools
    from keras import backend as K
    import tensorflow as tf
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper

batch_size = 512
num_classes = 2389
epochs = 12

# input image dimensions
img_rows, img_cols = 72, 72

train_dir = settings.TRAIN_DIR
test_dir = settings.TEST_DIR


#Using generators to simplify data preparations
test_datagen = ImageDataGenerator()
train_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size= (img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale' #probably should use grayscale mode
)

test_generator = train_datagen.flow_from_directory(
    test_dir,
    target_size= (img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale'  #probably should use grayscale mode
)
model = MnistCnnModel(num_classes, (img_rows, img_cols), 1).get_model()
# If we train this another time with same specs - uncomment this
model.load_weights('./checkpoint-04-0.88.hdf5')

checkpoint = ModelCheckpoint(settings.CHECKPOINT_PATTERN, monitor='acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

precision = as_keras_metric(tf.metrics.precision)
recall = as_keras_metric(tf.metrics.recall)
false_positives = as_keras_metric(tf.metrics.false_positives)
true_positives = as_keras_metric(tf.metrics.true_positives)

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit_generator(
        train_generator,
        steps_per_epoch=24, # Changeable
        epochs=25, # Changeable
        validation_data=test_generator,
        validation_steps=2, # Changeable
        callbacks=callbacks_list)
model.save_weights(settings.MODEL_NAME)
