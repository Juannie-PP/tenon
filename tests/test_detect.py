import os
import pytest
from src.tenon import rotate_identify, notch_identify

current_work_dir = os.path.dirname(__file__)


@pytest.mark.parametrize(
    "small_circle, big_circle, image_type, expected",
    [
        (
            current_work_dir + "/image/inner.png",
            current_work_dir + "/image/outer.png",
            2,
            311,
        ),
    ],
)
def test_rotate_identify(small_circle, big_circle, image_type, expected):
    result = rotate_identify(
        small_circle,
        big_circle,
        image_type,
    )

    assert result.total_rotate_angle == expected


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
