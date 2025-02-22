# Ray-Tracing
Ray Tracing Assignment for the course Advanced Computer Architecture

## Instructions to Run
For CPU Code:
>> g++ cpu_tracing.cpp -o cpu_tracing
>> cpu_tracing 800 10
>> python display.py cpu
For GPU Code:
>> nvcc gpu_tracing.cu -o gpu_tracing
>> gpu_tracing 800 10
>> python display.py gpu
