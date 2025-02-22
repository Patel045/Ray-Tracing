from PIL import Image
import numpy as np
import sys

argc = len(sys.argv)
if(argc != 2):
    print("Accepts only 1 argument")
    print("python display.py <cpu/gpu>")
    exit()

name = sys.argv[1] + "_image"

with open(name+".ppm", "rb") as f:
    f.readline()  # P3
    width, height = map(int, f.readline().split())
    f.readline()  # Max color (255)
    pixels = np.loadtxt(f, dtype=int).reshape(height, width, 3)

img = Image.fromarray(pixels.astype('uint8'), 'RGB')
img.save(name+".png")
img.show()
