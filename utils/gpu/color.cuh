#ifndef COLOR_CUH
#define COLOR_CUH

#include "vec3.cuh"

#define color vec3

__host__ __device__ int minn(int a, int b){
    if(a > b) return b;
    return a;
}

__host__ __device__ int maxx(int a, int b){
    if(a > b) return a;
    return b;
}

__host__ __device__
vec3 translate_color(const color& pixel_color){
    // Translate the [0,1] component values to the byte range [0,255].
    int rbyte = maxx(0,minn(255,int(255.999 * pixel_color.x())));
    int gbyte = maxx(0,minn(255,int(255.999 * pixel_color.y())));
    int bbyte = maxx(0,minn(255,int(255.999 * pixel_color.z())));

    return vec3(rbyte,gbyte,bbyte);
}

#endif