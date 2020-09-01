from PIL import Image
import numpy as np

sample_img_path = '../data/cat/cat1.jpeg'
sample_img_size = (200, 300)


def get_test_video_duplicated_image(clip_size):
    img = Image.open(sample_img_path)
    clip = [img] * clip_size
    return clip


def get_test_video_color_sequence(clip_size):
    colors = np.linspace(0, 255, clip_size).astype(int)
    clip = [color*np.ones(sample_img_size+(3,)) for color in colors]
    return clip
