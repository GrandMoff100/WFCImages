from collections import defaultdict
from email.generator import Generator
from email.policy import default
import itertools
import numpy as np

N = 3
OUTPUT_SIZE = 100

INPUT = """
0100
0100
0100
0100
0100
0100
1111
0100
"""

# Comvert it to a numpy array
img = np.array([[int(x) for x in row] for row in INPUT.strip().split("\n")])


def augment_tile(tile: np.ndarray) -> Generator[np.ndarray, None, None]:
    for _ in range(4):
        yield tile
        tile = np.rot90(tile)
    tile = np.fliplr(tile)
    for _ in range(4):
        yield tile
        tile = np.rot90(tile)


def get_tile_frequencies(img, tile_size) -> tuple[dict[int, int], dict[int, tuple[int, ...]]]:
    """Get each tile augmentation and its frequency."""
    frequencies: dict[tuple[int, ...], int] = defaultdict(int)
    for i, j in itertools.product(range(img.shape[0]), range(img.shape[1])):
        tile = img.take(range(i, i + tile_size), mode="wrap", axis=0).take(range(j, j + tile_size), mode="wrap", axis=1)
        for augmented_tile in augment_tile(tile):
            frequencies[tuple(augmented_tile.flatten())] += 1
    return dict(enumerate(frequencies.values())), dict(enumerate(frequencies.keys()))

