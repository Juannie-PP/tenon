import os
import pytest
from src.tenon import rotate_identify, notch_identify

current_work_dir = os.path.dirname(__file__)


@pytest.mark.parametrize(
    "small_circle, big_circle, image_type, small_circle_crop_pixel, expected",
    [
        (
            current_work_dir + "/image/inner.png",
            current_work_dir + "/image/outer.png",
            2,
            102,
            313,
        ),
    ],
)
def test_rotate_identify(
    small_circle, big_circle, image_type, small_circle_crop_pixel, expected
):
    result = rotate_identify(
        small_circle,
        big_circle,
        image_type,
        small_circle_crop_pixel=small_circle_crop_pixel,
    )

    assert result["total_angle"] == expected


@pytest.mark.parametrize(
    "slider, background, image_type, expected",
    [
        (
            current_work_dir + "/image/slide.png",
            current_work_dir + "/image/background.png",
            2,
            133,
        ),
    ],
)
def test_notch_identify(slider, background, image_type, expected):
    result = notch_identify(slider, background, image_type)
    assert result == expected
