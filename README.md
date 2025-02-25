# Ray-Tracing
Ray Tracing Assignment for the course Advanced Computer Architecture

## Instructions to Run
Make sure to install the python libraries - Numpy, PIL, sys before running the following commands.  
For CPU Code:
```
g++ cpu_tracing.cpp -o cpu_tracing
cpu_tracing <image width> <samples for anti-aliasing> <world.txt>
python display.py <world_render.ppm>
```
For GPU Code:
```
nvcc gpu_tracing.cu -o gpu_tracing
gpu_tracing <image width> <samples for anti-aliasing> <world.txt>
python display.py <world_render.ppm>
```

To play around with the world setup, change the values in the world.txt files. 

## Format for world.txt files
Each world.txt file consists of:
1. Camera definition (First line)
   - Camera Camera_Position(x y z) Camera_Direction(x y z) Upward_Direction(x y z) Field_Of_View
2. List of spheres (One per line).
   - Sphere_Name Center(x y z) Radius Reflect(True/False) (Color(r g b) if not reflective)
