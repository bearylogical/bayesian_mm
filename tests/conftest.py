import pytest
import json
from pathlib import Path
from typing import Tuple, List
from PIL import Image
import uuid

# from src.utils.shapes.circles import CirclesGenerator
from tempfile import TemporaryDirectory
import random

from keras.models import Input
from src.inference.data import generate_data_v2

from src.keypoint.models.regression.cnn_regression import BaseKeypointModel
from src.keypoint.models import KeypointDetector

SEED = 42


@pytest.fixture
def mock_data():
    num_obs = 10
    num_experiment = 1
    length_scale = 1e3
    material_params = {
        "G": 30000 / length_scale,
        "K": 30000 / length_scale,
        "length_scale": length_scale,
    }
    return generate_data_v2(
        num_obs=num_obs, num_experiments=num_experiment, **material_params
    )


@pytest.fixture
def mock_image():
    return Image.new("L", (128, 128), 1)


@pytest.fixture(scope="session")
def mock_disk_image(tmpdir_factory) -> Path:
    img = Image.new("L", (600, 600), 1)
    mock_path = tmpdir_factory.mktemp("data").join("mock_img.png")
    img.save(Path(mock_path))
    return mock_path


@pytest.fixture(scope="session")
def mock_bulk_image_dir(tmpdir_factory) -> Path:
    """
    Recursively create  image directories
    Parameters
    ----------
    tempdir_factory

    Returns
    -------

    """
    mock_dir = tmpdir_factory.mktemp("data").mkdir("test_bulk_gen")
    num_parents = 3
    max_child_per_parent = 10
    for parent in range(num_parents):
        parent_dir = mock_dir.mkdir(f"parent{parent}")
        # write some rubbish
        sub_dir_text = parent_dir.join("s.txt").write("test")
        for child_img in range(max_child_per_parent):
            img = Image.new("L", (600, 600), 1)
            img_path = parent_dir.join(f"{child_img}.png")
            img.save(Path(img_path))

    return Path(mock_dir)


@pytest.fixture(scope="session")
def mock_img_keypoint_label(tmpdir_factory) -> Tuple[Path, Path]:
    mock_img_dir = tmpdir_factory.mktemp("data").mkdir("test_img_keypoint_loader")
    mock_label_dir = mock_img_dir.mkdir("labels")
    mock_images = 10
    mock_keypoint_data = {
        "original_width": 600,
        "original_height": 600,
        "keypoints": {
            "p0": dict(x=1.0, y=41.0, width=0.0),
            "p1": dict(x=2.0, y=42.0, width=0.0),
            "p2": dict(x=3.0, y=43.0, width=0.0),
            "p3": dict(x=4.0, y=44.0, width=0.0),
            "p4": dict(x=5.0, y=45.0, width=0.0),
            "p5": dict(x=6.0, y=46.0, width=0.0),
            "p6": dict(x=7.0, y=47.0, width=0.0),
        },
    }

    for _ in range(mock_images):
        img = Image.new("L", (600, 600), 1)
        img_name = uuid.uuid4().hex[:10]
        img_path = Path(mock_img_dir.join(f"{img_name}.png"))
        img.save(img_path)
        mock_keypoint_data["file_name"] = img_path.name
        with open(mock_label_dir.join(f"{img_name}.json"), "w+") as mock_keypoint_file:
            json.dump(mock_keypoint_data, mock_keypoint_file)

    return Path(mock_img_dir), Path(mock_label_dir)


@pytest.fixture()
def create_temp_dir() -> Path:
    temp_dir_path = Path.cwd() / "temp"
    temp_dir_path = TemporaryDirectory(dir=temp_dir_path)
    yield Path(temp_dir_path.name)
    temp_dir_path.cleanup()


def mock_base_keypoint_model():
    base_model = BaseKeypointModel()
    input = Input((224, 224, 1))
    base_model.build(input_shape=(None, 224, 224, 1))
    base_model.call(input)

    return base_model


@pytest.fixture(scope="session")
def mock_keypoint_detector() -> KeypointDetector:

    keypoint_detector = KeypointDetector()
    keypoint_detector.model = mock_base_keypoint_model()
    return keypoint_detector


# @pytest.fixture
# def mock_circles_generator():
#     temp_dir = Path.cwd() / 'dataset'
#     temp_dir = TemporaryDirectory(dir=temp_dir)
#     yield CirclesGenerator(n_images=1, max_circles=1, save_dir=Path(temp_dir.name), seed=SEED)
#     # destroy on completion
#     temp_dir.cleanup()
