
from __future__ import print_function

from keras.callbacks import ModelCheckpoint
from keras_preprocessing.image import ImageDataGenerator

import settings
from models import MnistCnnModel

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
    color_mode='rgb' #probably should use grayscale mode
)

test_generator = train_datagen.flow_from_directory(
    test_dir,
    target_size= (img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='rgb'  #probably should use grayscale mode
)
model = MnistCnnModel(num_classes, (img_rows, img_cols), 3).get_model()
# If we train this another time with same specs - uncomment this
# model.load_weights(settings.MODEL_NAME)

checkpoint = ModelCheckpoint(settings.CHECKPOINT_PATTERN, monitor='acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.compile(loss='categorical_crossentropy',
              optimizer='Adadelta',
              metrics=['accuracy'])

model.fit_generator(
        train_generator,
        steps_per_epoch=250, # Changeable
        epochs=20, # Changeable
        validation_data=test_generator,
        validation_steps=100, # Changeable
        callbacks=callbacks_list)
model.save_weights(settings.MODEL_NAME)
