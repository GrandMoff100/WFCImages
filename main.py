import numpy as np

from wfcimages import ImageProcessor

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
image = np.array([[int(x) for x in row] for row in INPUT.strip().split("\n")])

# Create an ImageProcessor
processor = ImageProcessor(image, N)
