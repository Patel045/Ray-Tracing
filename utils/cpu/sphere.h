#ifndef SPHERE_H
#define SPHERE_H

#include "vec3.h"
#include "color.h"
#include "ray.h"
#include "hit_data.h"

#define reflection_limit 1

color ray_color(const ray& r);

class Sphere{
public: 
    Sphere() = delete;
    Sphere(const point3& center, double radius)
        :center(center), radius(radius), color_val(-1,-1,-1), reflect(false){}
    Sphere(const point3& center, double radius, const color& color_val)
        :center(center), radius(radius), color_val(color_val), reflect(false){}
    Sphere(const point3& center, double radius, bool reflect)
        :center(center), radius(radius), color_val(0.8,0.8,0.8), reflect(reflect){}
    
    point3 center;
    double radius;
    color color_val;
    bool reflect;

    hit_data hit_sphere(const ray& r) {
        vec3 oc = center - r.origin();
        auto a = r.direction().magnitude_squared();
        auto h = dot(r.direction(), oc);
        auto c = oc.magnitude_squared() - radius*radius;
        auto discriminant = h*h - a*c;

        if (discriminant < 0) {
            return hit_data();
        } else {
            double t = (h - std::sqrt(discriminant)) / a;
            if(t < 0){
                return hit_data();
            }

            color hit_color = color_val;
            point3 poc = r.at(t);
            double hit_distance = (r.origin() - poc).magnitude();
            vec3 N = unit_vector(poc - center);

            if(hit_color.x() < 0){
                hit_color = 0.5*color(N.x()+1, N.y()+1, N.z()+1);
            }

            ray normal(poc,N);

            if(reflect && r.ref_ctr < reflection_limit){
                ray reflected(poc, r.direction() - 2*dot(r.direction(),normal.direction())*normal.direction(), r.ref_ctr+1); 
                hit_color = (ray_color(reflected)+ hit_color)/2.0;
            }
            
            return hit_data(hit_color, hit_distance, normal);
        }
    }


};

#endif