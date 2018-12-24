import Augmentor
import os
import shutil
import ujson

from PIL import Image


def augment_data(path: str):
    p = Augmentor.Pipeline(path, path)
    p.random_distortion(1, 4, 4, 4)
    p.skew_corner(0.5)
    p.sample(100)


def move_img_to_folders(path: str):
    for file in os.listdir(path):
        if file.endswith('.png'):
            new_folder = os.path.join(path, file).split('.')[0]
            os.makedirs(new_folder)
            shutil.move(os.path.join(path, file), new_folder)


def augment_emojis(path: str):
    for dir in os.listdir(path):
        path_to_dir = os.path.join(path, dir)
        augment_data(path_to_dir)

def transform_to_jpeg(path: str):
    for dir in os.listdir(path):
        path_to_dir = os.path.join(path, dir)
        for image in os.listdir(path_to_dir):
            im = Image.open(os.path.join(path_to_dir, image))
            rgb_im = im.convert('RGB')
            rgb_im.save(os.path.join(path_to_dir, image).split('.')[0] + ".jpg")
            print(f'saved: {os.path.join(path_to_dir, image).split(".")[0] + ".jpg"}')


def rename_folders(path: str, json_path: str):
    with open(json_path) as f:
        data = ujson.load(f)
        for emoji in data:
            try:
                os.rename(os.path.join(path, emoji['unicode']), os.path.join(path, emoji['description']))
                print(f'renamed {emoji["unicode"]} to {emoji["description"]}')
            except Exception as ex:
                print(f'dir has been already renamed: {emoji["unicode"]}')

def get_min(path):
    min = 200
    for dir in os.listdir(path):
        path_to_dir = os.path.join(path, dir)
        print(len(os.listdir(path_to_dir)))
        if len(os.listdir(path_to_dir)) < min:
            min = len(os.listdir(path_to_dir))
    print(min)

def enumerate_classes(path: str):
    sorted_dir = sorted(os.listdir(path))
    with open('./classes.json', 'w+') as f:
        ujson.dump({sorted_dir.index(x): x for x in sorted_dir}, f)


if __name__ == '__main__':
    # need to use one time for futher data augmentation
    move_img_to_folders('/home/atticus/emoji-dataset/emoji_imgs_V5')
    #
    # # than apply augmentation!
    augment_emojis('/home/atticus/emojis/train')
    #
    # # rename folder to emoji description
    rename_folders('/home/atticus/emoji-dataset/emoji_imgs_V5', '/home/atticus/emojis/V5_data.json')
    enumerate_classes('/home/atticus/emojis/train')


