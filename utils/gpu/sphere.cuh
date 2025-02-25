#ifndef SPHERE_CUH
#define SPHERE_CUH

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include "vec3.cuh"
#include "color.cuh"
#include "ray.cuh"
#include "hit_data.cuh"

#define reflection_limit 1

class Sphere;

__host__ __device__
color ray_color(const ray& r, Sphere* spheres, int num);

class Sphere{
public: 
    __host__ __device__ Sphere() = delete;    
    __host__ __device__ Sphere(const point3& center, double radius, bool reflect)
        :center(center),radius(radius), reflect(reflect), color_val(0.8,0.8,0.8){}
    __host__ __device__ Sphere(const point3& center, double radius, bool reflect, const color& color_val)
        :center(center),radius(radius),reflect(reflect), color_val(color_val){}

    point3 center;
    double radius;
    bool reflect;
    color color_val;

    __host__ __device__ hit_data hit_sphere(const ray& r, Sphere* spheres, int num) {
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
                hit_color = (ray_color(reflected,spheres,num)+ hit_color)/2.0;
            }
            
            return hit_data(hit_color, hit_distance, normal);
        }
    }
};

__host__ 
void loadSpheresFromFile(const std::string& filename, std::vector<Sphere>& spheres) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Error opening file: " << filename << "\n";
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string name;
        if(!(iss >> name )) continue;

        double x, y, z, r;
        if(!(iss >> x >> y >> z >> r)) continue;
        
        std::string reflect;
        if(!(iss >> reflect)) continue;
        bool reflect_b = (reflect == "true");

        if(reflect_b){
            spheres.emplace_back(point3(x,y,z),r,reflect_b);
        }
        else{
            double cr, cg, cb;
            if(!(iss >> cr >> cg >> cb)) continue;
            spheres.emplace_back(point3(x,y,z),r,reflect_b,color(cr,cg,cb));
        }
    }
}

#endif