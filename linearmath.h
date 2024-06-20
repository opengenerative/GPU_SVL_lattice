/*
    Reference  - https://github.com/NVIDIA/cuda-samples/blob/master/Samples/5_Domain_Specific/simpleVulkan/linmath.h
*/

#pragma once

#ifndef LINMATH_H
#define LINMATH_H


# define M_PI		3.14159265358979323846	

// Converts degrees to radians.
#define degreesToRadians(angleDegrees) (angleDegrees * M_PI / 180.0)

// Converts radians to degrees.
#define radiansToDegrees(angleRadians) (angleRadians * 180.0 / M_PI)

typedef float vec3[3];

static inline void vec3_sub(vec3 r, vec3 const a, vec3 const b)
{
    int i;
    for (i = 0; i < 3; ++i) r[i] = a[i] - b[i];
}

static inline void vec3_scale(vec3 r, vec3 const v, float const s) 
{
    int i;
    for (i = 0; i < 3; ++i) r[i] = v[i] * s;
}

static inline float vec3_mul_inner(vec3 const a, vec3 const b) 
{
    float p = 0.f;
    int i;
    for (i = 0; i < 3; ++i) p += b[i] * a[i];
    return p;
}

static inline void vec3_mul_cross(vec3 r, vec3 const a, vec3 const b) 
{
    r[0] = a[1] * b[2] - a[2] * b[1];
    r[1] = a[2] * b[0] - a[0] * b[2];
    r[2] = a[0] * b[1] - a[1] * b[0];
}

static inline float vec3_len(vec3 const v) { return sqrtf(vec3_mul_inner(v, v)); }

static inline void vec3_norm(vec3 r, vec3 const v) 
{
    float k = 1.f / vec3_len(v);
    vec3_scale(r, v, k);
}

typedef float vec4[4];

static inline float vec4_mul_inner(vec4 a, vec4 b)
{
    float p = 0.f;
    int i;
    for (i = 0; i < 4; ++i) p += b[i] * a[i];
    return p;
}

typedef vec4 mat4x4[4];

static inline void mat4x4_row(vec4 r, mat4x4 M, int i) 
{
    int k;
    for (k = 0; k < 4; ++k) r[k] = M[k][i];
}

static inline void mat4x4_mul(mat4x4 M, mat4x4 a, mat4x4 b) 
{
    int k, r, c;
    for (c = 0; c < 4; ++c)
        for (r = 0; r < 4; ++r) {
            M[c][r] = 0.f;
            for (k = 0; k < 4; ++k) M[c][r] += a[k][r] * b[c][k];
        }
}

static inline void mat4x4_translate_in_place(mat4x4 M, float x, float y, float z) 
{
    vec4 t = {x, y, z, 0};
    vec4 r;
    int i;
    for (i = 0; i < 4; ++i) {
        mat4x4_row(r, M, i);
        M[3][i] += vec4_mul_inner(r, t);
    }
}



static inline void mat4x4_ortho(mat4x4 M, float l, float r, float b, float t, float n, float f) 
{
    M[0][0] = 2.f / (r - l);
    M[0][1] = M[0][2] = M[0][3] = 0.f;

    M[1][1] = 2.f / (t - b);
    M[1][0] = M[1][2] = M[1][3] = 0.f;

    M[2][2] = -2.f / (f - n);
    M[2][0] = M[2][1] = M[2][3] = 0.f;

    M[3][0] = -(r + l) / (r - l);
    M[3][1] = -(t + b) / (t - b);
    M[3][2] = -(f + n) / (f - n);
    M[3][3] = 1.f;
}

static inline void mat4x4_perspective(mat4x4 m, float y_fov, float aspect, float n, float f) {

    float const a = (float)(1.f / tan(y_fov / 2.f));

    m[0][0] = a / aspect;
    m[0][1] = 0.f;
    m[0][2] = 0.f;
    m[0][3] = 0.f;

    m[1][0] = 0.f;
    m[1][1] = a;
    m[1][2] = 0.f;
    m[1][3] = 0.f;

    m[2][0] = 0.f;
    m[2][1] = 0.f;
    m[2][2] = -((f + n) / (f - n));
    m[2][3] = -1.f;

    m[3][0] = 0.f;
    m[3][1] = 0.f;
    m[3][2] = -((2.f * f * n) / (f - n));
    m[3][3] = 0.f;
}


static inline void mat4x4_look_at(mat4x4 m, vec3 eye, vec3 center, vec3 up) 
{

    vec3 f;
    vec3_sub(f, center, eye);
    vec3_norm(f, f);

    vec3 s;
    vec3_mul_cross(s, f, up);
    vec3_norm(s, s);

    vec3 t;
    vec3_mul_cross(t, s, f);

    m[0][0] = s[0];
    m[0][1] = t[0];
    m[0][2] = -f[0];
    m[0][3] = 0.f;

    m[1][0] = s[1];
    m[1][1] = t[1];
    m[1][2] = -f[1];
    m[1][3] = 0.f;

    m[2][0] = s[2];
    m[2][1] = t[2];
    m[2][2] = -f[2];
    m[2][3] = 0.f;

    m[3][0] = 0.f;
    m[3][1] = 0.f;
    m[3][2] = 0.f;
    m[3][3] = 1.f;

    mat4x4_translate_in_place(m, -eye[0], -eye[1], -eye[2]);
}

#endif
