#include "utils/cpu/color.h"
#include "utils/cpu/ray.h"
#include "utils/cpu/vec3.h"
#include "utils/cpu/hit_data.h"
#include "utils/cpu/sphere.h"

#include <bits/stdc++.h>
#include <chrono>
#include <random>

using namespace std::chrono;

color ray_color(const ray& r) {
    //default sky
    vec3 unit_direction = unit_vector(r.direction());
    auto a = 0.5*(unit_direction.y() + 1.0);
    color render_color =  (1.0-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);
    double cur_distance = inf;

    //Set the Stage !!
    static std::vector<Sphere> spheres = {
        Sphere(point3(0,0,-1), 0.5, true),
        Sphere(point3(1,0,1), 0.45, color(0.9,0.9,0.05)),
        Sphere(point3(-1,0,-0.5), 0.25, color(1,0.1,0.1)),
        Sphere(point3(0.8,0.5,-1), 0.3),
        Sphere(point3(0,-100.5,-1), 100, color(0,1,0))  
    }; 

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

int main() {

    std::ofstream img_file("cpu_image.ppm");

    // Image
    auto aspect_ratio = 16.0 / 9.0;
    int image_width = 1000;
    int samples_per_pixel = 50;

    // Calculate the image height, and ensure that it's at least 1.
    int image_height = int(image_width / aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;

    // Camera
    auto focal_length = 1.0;
    auto viewport_height = 2.0;
    auto viewport_width = viewport_height * (double(image_width)/image_height);
    auto camera_center = point3(0, 0, 0);

    // Calculate the vectors across the horizontal and down the vertical viewport edges.
    auto viewport_u = vec3(viewport_width, 0, 0);
    auto viewport_v = vec3(0, -viewport_height, 0);

    // Calculate the horizontal and vertical delta vectors from pixel to pixel.
    auto pixel_delta_u = viewport_u / image_width;
    auto pixel_delta_v = viewport_v / image_height;

    // Calculate the location of the upper left pixel.
    auto viewport_upper_left = camera_center
                             - vec3(0, 0, focal_length) - viewport_u/2 - viewport_v/2;
    auto pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

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
    std::cout << " Time Taken : " << duration.count() << std::endl;

    for(int i=0; i<image_height*image_width; i++){
        img_file << output_color[i] << '\n';
    }

    std::clog << "\rDone.                 \n";
    return 0;
}