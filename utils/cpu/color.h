#ifndef COLOR_H
#define COLOR_H

#include "vec3.h"

#include <iostream>

using color = vec3;

vec3 translate_color(const color& pixel_color){
    auto r = pixel_color.x();
    auto g = pixel_color.y();
    auto b = pixel_color.z();

    // Translate the [0,1] component values to the byte range [0,255].
    int rbyte = std::max(0,std::min(255,int(255.999 * r)));
    int gbyte = std::max(0,std::min(255,int(255.999 * g)));
    int bbyte = std::max(0,std::min(255,int(255.999 * b)));

    return vec3(rbyte,gbyte,bbyte);
}

#endif