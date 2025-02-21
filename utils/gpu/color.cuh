#ifndef COLOR_CUH
#define COLOR_CUH

#include "vec3.cuh"

using color = vec3;

// __host__ __device__
// void write_color(std::ostream& out, const color& pixel_color) {
//     auto r = pixel_color.x();
//     auto g = pixel_color.y();
//     auto b = pixel_color.z();

//     // Translate the [0,1] component values to the byte range [0,255].
//     int rbyte = int(255.999 * r);
//     int gbyte = int(255.999 * g);
//     int bbyte = int(255.999 * b);

//     // Write out the pixel color components.
//     out << rbyte << ' ' << gbyte << ' ' << bbyte << '\n';
// }

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
    auto r = pixel_color.x();
    auto g = pixel_color.y();
    auto b = pixel_color.z();

    // Translate the [0,1] component values to the byte range [0,255].
    int rbyte = maxx(0,minn(255,int(255.999 * r)));
    int gbyte = maxx(0,minn(255,int(255.999 * g)));
    int bbyte = maxx(0,minn(255,int(255.999 * b)));

    return vec3(rbyte,gbyte,bbyte);
}

#endif