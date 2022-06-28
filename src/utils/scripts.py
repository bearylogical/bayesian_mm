from typing import List

import click

from src.utils.constants import ACCEPTABLE_IMAGE_FORMATS
from src.utils.converter import batch_rescale_dir


@click.command()
@click.argument('src_dir', nargs=-1, type=click.Path(exists=True))
@click.option('t', '--target_dir', required=False)
def cli_batch_rescale(src_dir: str,
                      target_dir: str = None,
                      file_format: List[str] = ACCEPTABLE_IMAGE_FORMATS):
    batch_rescale_dir(src_dir, target_dir, file_format)
