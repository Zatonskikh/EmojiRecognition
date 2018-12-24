import os

# Modify according to your system
MODEL_NAME = os.environ.get("MODEL_NAME", "grayscale.h5")
CHECKPOINT_PATTERN = os.environ.get("CHECKPOINT_PATTERN", "./checkpoint-{epoch:02d}-{val_acc:.2f}.hdf5")
TRAIN_DIR = os.environ.get("TRAIN_DIR", '/home/atticus/emojis/emoji_imgs_V5 (бекап с ренеймом)')
TEST_DIR = os.environ.get("TEST_DIR", '/home/atticus/emojis/test')
TEST_IMAGE_PATH = os.environ.get("TEST_IMAGE_PATH", '/home/atticus/nauseated-face_1f922 (1).png')
JSON_CLASSES = os.environ.get("JSON_CLASSES", './classes.json')
