import itertools
from collections import defaultdict
from typing import Generator

import numpy as np


class ImageProcessor:
    def __init__(self, img: np.ndarray, tile_size: int) -> None:
        self.img = img
        self.tile_size = tile_size
        self.tiles = self.get_tile_frequencies()

    def augment_tile(self, tile: np.ndarray) -> Generator[np.ndarray, None, None]:
        for _ in range(4):
            yield tile
            tile = np.rot90(tile)
        tile = np.fliplr(tile)
        for _ in range(4):
            yield tile
            tile = np.rot90(tile)

    def get_tile_frequencies(self) -> dict[int, tuple[tuple[int, ...], int]]:
        """Get each tile augmentation and its frequency."""
        frequencies: dict[tuple[int, ...], int] = defaultdict(int)
        for i, j in itertools.product(
            range(self.img.shape[0]), range(self.img.shape[1])
        ):
            tile = self.img.take(
                range(i, i + self.tile_size), mode="wrap", axis=0
            ).take(range(j, j + self.tile_size), mode="wrap", axis=1)
            for augmented_tile in self.augment_tile(tile):
                frequencies[tuple(augmented_tile.flatten())] += 1
        return dict(enumerate(frequencies.items()))
