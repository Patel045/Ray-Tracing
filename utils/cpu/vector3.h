#ifndef VECTOR3_H
#define VECTOR3_H

#include <cmath>
#include <iostream>

class Vector3 {
  public:
    double elements[3];

    Vector3() : elements{0, 0, 0} {}
    Vector3(double x, double y, double z) : elements{x, y, z} {}

    double getX() const { return elements[0]; }
    double getY() const { return elements[1]; }
    double getZ() const { return elements[2]; }

    Vector3 operator-() const { return Vector3(-elements[0], -elements[1], -elements[2]); }
    double operator[](int index) const { return elements[index]; }
    double& operator[](int index) { return elements[index]; }

    Vector3& operator+=(const Vector3& other) {
        elements[0] += other.elements[0];
        elements[1] += other.elements[1];
        elements[2] += other.elements[2];
        return *this;
    }

    Vector3& operator*=(double scalar) {
        elements[0] *= scalar;
        elements[1] *= scalar;
        elements[2] *= scalar;
        return *this;
    }

    Vector3& operator/=(double scalar) {
        return *this *= (1 / scalar);
    }

    double magnitude() const {
        return std::sqrt(magnitudeSquared());
    }

    double magnitudeSquared() const {
        return elements[0] * elements[0] + elements[1] * elements[1] + elements[2] * elements[2];
    }
};

using Point3 = Vector3;

// Vector Operations

inline std::ostream& operator<<(std::ostream& output, const Vector3& vec) {
    return output << vec.elements[0] << ' ' << vec.elements[1] << ' ' << vec.elements[2];
}

inline Vector3 operator+(const Vector3& a, const Vector3& b) {
    return Vector3(a.elements[0] + b.elements[0], a.elements[1] + b.elements[1], a.elements[2] + b.elements[2]);
}

inline Vector3 operator-(const Vector3& a, const Vector3& b) {
    return Vector3(a.elements[0] - b.elements[0], a.elements[1] - b.elements[1], a.elements[2] - b.elements[2]);
}

inline Vector3 operator*(const Vector3& a, const Vector3& b) {
    return Vector3(a.elements[0] * b.elements[0], a.elements[1] * b.elements[1], a.elements[2] * b.elements[2]);
}

inline Vector3 operator*(double scalar, const Vector3& vec) {
    return Vector3(scalar * vec.elements[0], scalar * vec.elements[1], scalar * vec.elements[2]);
}

inline Vector3 operator*(const Vector3& vec, double scalar) {
    return scalar * vec;
}

inline Vector3 operator/(const Vector3& vec, double scalar) {
    return (1 / scalar) * vec;
}

inline double dotProduct(const Vector3& a, const Vector3& b) {
    return a.elements[0] * b.elements[0] + a.elements[1] * b.elements[1] + a.elements[2] * b.elements[2];
}

inline Vector3 crossProduct(const Vector3& a, const Vector3& b) {
    return Vector3(a.elements[1] * b.elements[2] - a.elements[2] * b.elements[1],
                   a.elements[2] * b.elements[0] - a.elements[0] * b.elements[2],
                   a.elements[0] * b.elements[1] - a.elements[1] * b.elements[0]);
}

inline Vector3 normalize(const Vector3& vec) {
    return vec / vec.magnitude();
}

#endif
