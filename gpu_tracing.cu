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

#define PI 3.14159265359

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

__device__
color ray_color(const ray& r, Sphere* spheres, int num) {
    //default sky
    vec3 unit_direction = unit_vector(r.direction());
    auto a = 0.5*(unit_direction.y() + 1.0);
    color render_color =  (1.0-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);
    double cur_distance = inf;

    //Spheres
    for (int sp=0; sp<num; sp++){
        auto hit = spheres[sp].hit_sphere(r,spheres,num); 
        if(hit.hit && hit.hit_distance < cur_distance){
            cur_distance = hit.hit_distance;
            render_color = hit.hit_color;
        }
    }

    return render_color;
}

__global__
void dkernel(int image_height, int image_width, vec3 pixel00_loc, vec3 camera_center, vec3 pixel_delta_u, vec3 pixel_delta_v, color* color_data, Sphere* spheres, int num){
    int i = blockIdx.x % image_width;
    int j = blockIdx.x / image_width;
    int k = threadIdx.x;
    extern __shared__ color color_val[];       ////NOTE////
    __syncthreads();
    auto offset = sample_square(k);
    auto pixel_center = pixel00_loc + ((i + offset.x()) * pixel_delta_u) + ((j + offset.y()) * pixel_delta_v);
    auto ray_direction = pixel_center - camera_center;
    ray r(camera_center, ray_direction);
    
    color_val[k] = ray_color(r,spheres,num);
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

__host__
double dtr(double degrees) {
    return degrees * (PI / 180.0);
}

int main(int argc, char* argv[]){

    if(argc != 4){
        std::cout << "Accepts only 3 arguments" << std::endl;
        std::cout << "gpu_tracing <image_width> <samples_per_pixel> <textfile.txt>" << std::endl;
        return 0;
    }
    
    auto start_main = high_resolution_clock::now();
    
    std::string world_txt = argv[3];

    std::string world_ppm = world_txt.substr(0,world_txt.size()-4) + "_gpu.ppm";
    std::ofstream img_file(world_ppm);
    
    // Image
    int image_width = std::atoi(argv[1]);
    int samples_per_pixel = std::atoi(argv[2]);
    auto aspect_ratio = 16.0 / 9.0;
    int image_height = int(image_width / aspect_ratio);
    
    // Camera Parameters Declerartion
    point3 lookfrom, lookat;
    vec3 vup;
    double vfov;

    // Load the World
    std::vector<Sphere> spheres_v;
    loadWorldFromFile(world_txt,spheres_v,lookfrom,lookat,vup,vfov);
    Sphere* spheres_h = &spheres_v[0];
    Sphere* spheres;
    cudaMalloc(&spheres,sizeof(Sphere)*spheres_v.size());
    cudaMemcpy(spheres,spheres_h,sizeof(Sphere)*spheres_v.size(),cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // Camera Parameters Definition
    auto h = std::tan(dtr(vfov)/2);
    auto camera_center = lookfrom;
    auto focal_length = (lookfrom - lookat).magnitude();
    auto viewport_height = 2 * h * focal_length;
    auto viewport_width = viewport_height * (double(image_width)/image_height);

    // Viewport
    auto w = unit_vector(lookfrom - lookat);
    auto u = unit_vector(cross(vup, w));
    auto v = cross(w,u);
    auto viewport_u = viewport_width * u;
    auto viewport_v = -viewport_height * v;
    auto pixel_delta_u = viewport_u / image_width;
    auto pixel_delta_v = viewport_v / image_height;

    auto viewport_upper_left = camera_center - focal_length*w - viewport_u/2 - viewport_v/2;
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
        output_color_device,
        spheres,
        int(spheres_v.size())
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