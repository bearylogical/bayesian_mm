import pytest
from pathlib import Path
from PIL import Image
from src.utils.shapes import CirclesGenerator
from tempfile import TemporaryDirectory

SEED = 42


@pytest.fixture
def mock_image():
    return Image.new('L', (128, 128), 1)


@pytest.fixture
def mock_circles_generator():
    temp_dir = Path.cwd() / 'dataset'
    temp_dir = TemporaryDirectory(dir=temp_dir)
    yield CirclesGenerator(n_images=1, max_circles=1, save_dir=Path(temp_dir.name), seed=SEED)
    # destroy on completion
    temp_dir.cleanup()
