#ifndef HIT_DATA_CUH
#define HIT_DATA_CUH

#include "vec3.cuh"
#include "color.cuh"
#include "ray.cuh"

#define inf 1.7976931348623158e+308

class hit_data{
public:
    bool hit;
    color hit_color;
    double hit_distance;
    ray normal;

    __host__ __device__ hit_data()
        :hit(false),hit_color(-1,-1,-1),hit_distance(inf){};

    __host__ __device__ hit_data(const color& hit_color, double hit_distance, const ray& normal)
        :hit(true), hit_color(hit_color), hit_distance(hit_distance), normal(normal){};

    
};

#endif