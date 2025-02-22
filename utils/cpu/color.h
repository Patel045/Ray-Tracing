#ifndef COLOR_H
#define COLOR_H

#include "vec3.h"

#include <iostream>

#define color vec3

vec3 translate_color(const color & pixel_color){
    // Translate the [0,1] component values to the byte range [0,255].
    int rbyte = std::max(0,std::min(255,int(255.999 * pixel_color.x())));
    int gbyte = std::max(0,std::min(255,int(255.999 * pixel_color.y())));
    int bbyte = std::max(0,std::min(255,int(255.999 * pixel_color.z())));

    return vec3(rbyte,gbyte,bbyte);
}

#endif