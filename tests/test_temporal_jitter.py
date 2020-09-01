from tests.utils import get_test_video_color_sequence, sample_img_size
from torchvideotransforms.video_transforms import TemporalJitter, Compose
from torchvideotransforms.volume_transforms import ClipToTensor

channel_nb = 3


def test_temporal_jitter():
    clip = get_test_video_color_sequence(10)
    temp_jitter_transform = TemporalJitter(n_frames=(5,5))
    video_transform = Compose([
        temp_jitter_transform,
        ClipToTensor(channel_nb=channel_nb),
    ])
    tensor_clip = video_transform(clip)

    assert tensor_clip.shape == (3, 5) + sample_img_size