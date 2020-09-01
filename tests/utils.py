# Create dummy video clip
from PIL import Image

sample_img_path = '../data/cat/cat1.jpeg'
sample_img_size = (200, 300)


def get_test_video(clip_size):
    img = Image.open(sample_img_path)
    clip = [img] * clip_size
    return clip
