from PIL import Image
import numpy as np

inp = input()

name = inp + "_image"

with open(name+".ppm", "rb") as f:
    f.readline()  # P3
    width, height = map(int, f.readline().split())
    f.readline()  # Max color (255)
    pixels = np.loadtxt(f, dtype=int).reshape(height, width, 3)

img = Image.fromarray(pixels.astype('uint8'), 'RGB')
img.save(name+".png")
img.show()
