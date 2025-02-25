#include "utils/cpu/color.h"
#include "utils/cpu/ray.h"
#include "utils/cpu/vec3.h"
#include "utils/cpu/hit_data.h"
#include "utils/cpu/sphere.h"

#include <bits/stdc++.h>
#include <chrono>
#include <random>

using namespace std::chrono;

#define PI 3.14159265359

std::vector<Sphere> spheres;

color ray_color(const ray& r) {
    //default sky
    vec3 unit_direction = unit_vector(r.direction());
    auto a = 0.5*(unit_direction.y() + 1.0);
    color render_color =  (1.0-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);
    double cur_distance = inf;

    //Spheres
    for (auto s: spheres){
        auto hit = s.hit_sphere(r); 
        if(hit.hit && hit.hit_distance < cur_distance){
            cur_distance = hit.hit_distance;
            render_color = hit.hit_color;
        }
    }

    return render_color;
}

inline double random_double() {
    static std::uniform_real_distribution<double> distribution(0.0, 1.0);
    static std::mt19937 generator;
    return distribution(generator);
}

vec3 sample_square(){
    // Returns the vector to a random point in the [-.5,-.5]-[+.5,+.5] unit square.
    return vec3(random_double() - 0.5, random_double() - 0.5, 0);
}

double dtr(double degrees) {
    return degrees * (PI / 180.0);
}

int main(int argc, char* argv[]){

    if(argc != 4){
        std::cout << "Accepts only 3 arguments" << std::endl;
        std::cout << "cpu_tracing <image_width> <samples_per_pixel> <textfile.txt>" << std::endl;
        return 0;
    }

    auto start_main = high_resolution_clock::now();

    std::string world_txt = argv[3];

    std::string world_ppm = world_txt.substr(0,world_txt.size()-4) + "_cpu.ppm";
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
    loadWorldFromFile(world_txt,spheres,lookfrom,lookat,vup,vfov);

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
    color* output_color = new color[image_height*image_width];
    img_file << "P3\n" << image_width << " " << image_height << "\n255\n";
    
    // Render
    auto start = high_resolution_clock::now();

    for (int j = 0; j < image_height; j++) {
        for (int i = 0; i < image_width; i++) {
            color pixel_color(0,0,0);
            for (int sample = 0; sample<samples_per_pixel; sample++){
                auto offset = sample_square();
                auto pixel_center = pixel00_loc + ((i + offset.x()) * pixel_delta_u) + ((j + offset.y()) * pixel_delta_v);
                auto ray_direction = pixel_center - camera_center;
                ray r(camera_center, ray_direction);

                pixel_color += ray_color(r);
            }
            pixel_color /= samples_per_pixel;
            output_color[j*image_width+i] = translate_color(pixel_color);
        }
    }

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << " Computation Time Taken : " << duration.count() << std::endl;

    for(int i=0; i<image_height*image_width; i++){
        img_file << output_color[i] << '\n';
    }

    delete[] output_color;

    auto stop_main = high_resolution_clock::now();
    auto duration_main = duration_cast<microseconds>(stop_main - start_main);
    std::cout << " Total Time Taken : " << duration_main.count() << std::endl;

    std::clog << "\rDone.                 \n";
    return 0;
}