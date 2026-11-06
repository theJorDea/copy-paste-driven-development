import pytest

from scripts.prepare_yolo_dataset import convert_bbox


def test_convert_bbox_inside_image():
    result = convert_bbox(
        x=10,
        y=10,
        w=20,
        h=20,
        image_width=100,
        image_height=100,
    )

    assert result == pytest.approx((0.2, 0.2, 0.2, 0.2))


def test_convert_bbox_right_boundary():
    result = convert_bbox(
        x=90,
        y=10,
        w=20,
        h=20,
        image_width=100,
        image_height=100,
    )

    assert result == pytest.approx((0.95, 0.2, 0.1, 0.2))


def test_convert_bbox_left_boundary():
    result = convert_bbox(
        x=-10,
        y=10,
        w=30,
        h=20,
        image_width=100,
        image_height=100,
    )

    assert result == pytest.approx((0.1, 0.2, 0.2, 0.2))


def test_convert_bbox_outside_image():
    result = convert_bbox(
        x=110,
        y=10,
        w=20,
        h=20,
        image_width=100,
        image_height=100,
    )

    assert result is None
