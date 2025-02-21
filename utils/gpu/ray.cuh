#ifndef RAY_CUH
#define RAY_CUH

#include "vec3.cuh"

class ray {
  public:
    __host__ __device__ ray() {}

    __host__ __device__ ray(const point3& origin, const vec3& direction) : orig(origin), dir(direction), ref_ctr(0) {}
    __host__ __device__ ray(const point3& origin, const vec3& direction, int count) : orig(origin), dir(direction), ref_ctr(count) {}

    __host__ __device__ const point3& origin() const  { return orig; }
    __host__ __device__ const vec3& direction() const { return dir; }
    
    __host__ __device__ point3 at(double t) const {
      return orig + t*dir;
    }
    int ref_ctr;

  private:
    point3 orig;
    vec3 dir;
};

#endif