import ujson
import argparse
import numpy as np
from PIL import Image
from keras_preprocessing import image
from models import MnistCnnModel
import settings

num_classes = 2389
img_width, img_height = 72, 72

def predict(path_to_image: str):
    # load the model we saved
    model = MnistCnnModel(num_classes, (img_width, img_height), 1, './grayscale.h5').get_trained_model()
    img = image.load_img(path_to_image, target_size=(img_height, img_width),
                         color_mode='grayscale')
    # img.load()
    # background = Image.new('RGBA', (72, 72), (71, 113, 77, 255))
    # background.paste(img, mask=img.split()[3])  # 3 is the alpha channel
    image.save_img('./temp.png', image.img_to_array(img), file_format='png')
    # # background.convert('RGB') - doesn't work here but works in load_img
    #
    # x = image.load_img('./temp.png', color_mode='rgb')
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict_classes(images, batch_size=1)

    # print the classes, the images belong to
    with open(settings.JSON_CLASSES, 'r') as f:
        classes_dict = ujson.load(f)
        print(f'classes is: {classes_dict[str(classes[0])]}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image')
    args = parser.parse_args()
    predict(settings.TEST_IMAGE_PATH if args.image is None else args.image)