#ifndef VEC3_H
#define VEC3_H

#include <cmath>
#include <iostream>

#define point3 vec3

class vec3 {
  public:
    double arr[3];

    vec3() : arr{0,0,0} {}
    vec3(double a0, double a1, double a2) : arr{a0, a1, a2} {}
    ~vec3() {}

    double magnitude() const {
        return std::sqrt(magnitude_squared());
    }

    double magnitude_squared() const {
        return arr[0]*arr[0] + arr[1]*arr[1] + arr[2]*arr[2];
    }

    vec3& operator+=(const vec3& vec) {
        for (int i = 0; i < 3; ++i) arr[i] += vec.arr[i];
        return *this;
    }

    vec3& operator/=(double scalar) {
        for (int i = 0; i < 3; ++i) arr[i] /= scalar;
        return *this;
    }

    double x() const { return arr[0]; }
    double y() const { return arr[1]; }
    double z() const { return arr[2]; }
};

inline vec3 operator+(const vec3& u, const vec3& v) {
    return vec3(u.arr[0] + v.arr[0], u.arr[1] + v.arr[1], u.arr[2] + v.arr[2]);
}

inline vec3 operator-(const vec3& u, const vec3& v) {
    return vec3(u.arr[0] - v.arr[0], u.arr[1] - v.arr[1], u.arr[2] - v.arr[2]);
}

inline vec3 operator*(double scalar, const vec3& v) {
    return vec3(scalar*v.arr[0], scalar*v.arr[1], scalar*v.arr[2]);
}

inline vec3 operator/(const vec3& v, double scalar) {
    return (1/scalar) * v;
}

inline double dot(const vec3& u, const vec3& v) {
    return u.arr[0] * v.arr[0] + u.arr[1] * v.arr[1] + u.arr[2] * v.arr[2];
}

inline vec3 unit_vector(const vec3& v) {
    return v / v.magnitude();
}

inline std::ostream& operator<<(std::ostream& out, const vec3& v) {
    return out << v.arr[0] << ' ' << v.arr[1] << ' ' << v.arr[2];
}

#endif