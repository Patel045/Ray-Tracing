# Ray-Tracing
Ray Tracing Assignment for the course Advanced Computer Architecture

## Instructions to Run
For CPU Code:
```
g++ cpu_tracing.cpp -o cpu_tracing
cpu_tracing <image width> <samples for anti-aliasing>
python display.py cpu
```
For GPU Code:
```
nvcc gpu_tracing.cu -o gpu_tracing
gpu_tracing <image width> <samples for anti-aliasing>
python display.py gpu
```

To play around with the world setup, add or remove spheres in the ray_color()
function in both files, and compile & execute the codes.
