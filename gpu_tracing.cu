#include <chrono>
#include <fstream>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>

#include "utils/gpu/vec3.cuh"
#include "utils/gpu/color.cuh"
#include "utils/gpu/ray.cuh"
#include "utils/gpu/sphere.cuh"
#include "utils/gpu/hit_data.cuh"

using std::cin;
using std::cout;
using namespace std::chrono;


#define BLOCK_ID (blockIdx.x * gridDim.y * gridDim.z + blockIdx.y * gridDim.z + blockIdx.z)
#define THREAD_TOTAL (blockDim.x * blockDim.y * blockDim.z) 
#define THREAD_ID (threadIdx.x * blockDim.y * blockDim.z + threadIdx.y * blockDim.z + threadIdx.z)

__device__ 
vec3 sample_square(int idx){
    // Initialize cuRAND state with thread-specific values
    curandState state;
    curand_init(clock64() + idx, 0, 0, &state); // Using clock64() for randomness
    
    // Generate random float in range [0,1)
    double a1 = curand_uniform(&state) - 0.5;
    double a2 = curand_uniform(&state) - 0.5;
    return vec3(a1,a2,0);

}

__host__ __device__
color ray_color(const ray& r) {
    //default sky
    vec3 unit_direction = unit_vector(r.direction());
    auto a = 0.5*(unit_direction.y() + 1.0);
    color render_color =  (1.0-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);
    double cur_distance = inf;

    Sphere spheres[] = {
        Sphere(point3(0,0,-1), 0.5, true),
        Sphere(point3(1,0,1), 0.45, color(0.9,0.9,0.05)),
        Sphere(point3(-1,0,-0.5), 0.25, color(1,0.1,0.1)),
        Sphere(point3(0.8,0.5,-1), 0.3),
        Sphere(point3(0,-100.5,-1), 100, color(0,1,0))  
    }; 

    //Spheres
    for (int sp=0; sp<sizeof(spheres)/sizeof(Sphere); sp++){
        auto hit = spheres[sp].hit_sphere(r); 
        if(hit.hit && hit.hit_distance < cur_distance){
            cur_distance = hit.hit_distance;
            render_color = hit.hit_color;
        }
    }

    return render_color;
}

__global__
void dkernel(int image_height, int image_width, vec3 pixel00_loc, vec3 camera_center, vec3 pixel_delta_u, vec3 pixel_delta_v, color* color_data){
    int i = blockIdx.x % image_width;
    int j = blockIdx.x / image_width;
    int k = threadIdx.x;
    extern __shared__ color color_val[];       ////NOTE////
    __syncthreads();
    auto offset = sample_square(k);
    auto pixel_center = pixel00_loc + ((i + offset.x()) * pixel_delta_u) + ((j + offset.y()) * pixel_delta_v);
    auto ray_direction = pixel_center - camera_center;
    ray r(camera_center, ray_direction);
    
    color_val[k] = ray_color(r);
    __syncthreads();
    
    
    for(int off = (blockDim.x+1)/2; off >= 1; off = (off+1)/2){
        if(k < off){
            color_val[k] += color_val[k + off];
            color_val[k + off] = color();
        }
        __syncthreads();
        if(off == 1) break;
    }
    
    if(k==0){
        color_data[blockIdx.x] = translate_color(color_val[0]/(double)blockDim.x);
    }
    
}

int main(int argc, char* argv[]){

    if(argc != 3){
        std::cout << "Accepts only 2 arguments" << std::endl;
        std::cout << "gpu_tracing <image_width> <samples_per_pixel>" << std::endl;
        return 0;
    }
    
    auto start_main = high_resolution_clock::now();
    
    std::ofstream img_file("gpu_image.ppm");
    
    // Image
    int image_width = std::atoi(argv[1]);
    int samples_per_pixel = std::atoi(argv[2]);
    auto aspect_ratio = 16.0 / 9.0;
    int image_height = int(image_width / aspect_ratio);
    
    // Camera
    auto camera_center = point3(0, 0, 0);
    auto viewport_height = 2.0;
    auto viewport_width = viewport_height * (double(image_width)/image_height);
    auto focal_length = 1.0;
    
    // Viewport
    auto viewport_u = vec3(viewport_width, 0, 0);
    auto viewport_v = vec3(0, -viewport_height, 0);
    auto pixel_delta_u = viewport_u / image_width;
    auto pixel_delta_v = viewport_v / image_height;
    
    auto viewport_upper_left = camera_center - vec3(0, 0, focal_length) - viewport_u/2 - viewport_v/2;
    auto pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
    
    // Output
    img_file << "P3\n" << image_width << " " << image_height << "\n255\n";
    
    // Render
    auto start = high_resolution_clock::now();    

    ////////////////// CUDA //////////////////////
    int blocks_per_grid = image_height*image_width;
    int threads_per_block = samples_per_pixel;
    
    color* output_color = new color[image_height*image_width];
    color* output_color_device;
    cudaMalloc(&output_color_device, sizeof(color)*image_height*image_width);
    
    dkernel<<<blocks_per_grid,threads_per_block,sizeof(color)*(samples_per_pixel)>>>(
        image_height,
        image_width,
        pixel00_loc,
        camera_center,
        pixel_delta_u,
        pixel_delta_v,
        output_color_device
    );
    cudaDeviceSynchronize();
    
    cudaMemcpy(output_color,output_color_device,sizeof(color)*image_height*image_width,cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    ////////////////////////////////////////
    
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << " Computation Time Taken : " << duration.count() << std::endl;
    
    for(int i=0; i<image_height*image_width; i++){
        img_file << output_color[i] << '\n';
    }

    delete[] output_color;
    cudaFree(output_color_device);

    auto stop_main = high_resolution_clock::now();
    auto duration_main = duration_cast<microseconds>(stop_main - start_main);
    std::cout << " Total Time Taken : " << duration_main.count() << std::endl;

    std::clog << "\rDone.                 \n";

    return 0;
}