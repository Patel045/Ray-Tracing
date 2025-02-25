# Ray-Tracing
Ray Tracing Assignment for the course Advanced Computer Architecture

## Instructions to Run
Make sure to install the python libraries - Numpy, PIL, sys before running the following commands.  
```
pip install numpy
pip install Pillow
```
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

Note :- 
**Camera_Position**: Where the camera is located.

**Camera_Direction**: The direction the camera is looking.

**Upward_Direction**: Defines the "up" direction for orientation. Generally set as (0,1,0)

**Field_Of_View**: The FOV angle in degrees.

**Sphere_Name**: Any unique identifier (not used in rendering, but useful for organization).

**Center**(x y z): The sphere’s position in 3D space.

**Radius**: Sphere size.

**Reflect**: true for mirror-like surfaces, false for colored surfaces.

**Color**(r g b): If the sphere is not reflective, define its RGB color (values **between 0-1**).

For reflective surface color is hardcoded to be metallic-whitish. 

Color (-1,-1,-1) is a special color code for a sphere with a gradient of colors throught out its surface. 
