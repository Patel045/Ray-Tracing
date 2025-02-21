#ifndef RAY_H
#define RAY_H

#include "vec3.h"

class ray {
  public:
    ray() {}

    ray(const point3& origin, const vec3& direction) : orig(origin), dir(direction), ref_ctr(0) {}
    ray(const point3& origin, const vec3& direction, int count) : orig(origin), dir(direction), ref_ctr(count) {}

    const point3& origin() const  { return orig; }
    const vec3& direction() const { return dir; }
    int ref_ctr;

    point3 at(double t) const {
        return orig + t*dir;
    }

  private:
    point3 orig;
    vec3 dir;
};

#endif