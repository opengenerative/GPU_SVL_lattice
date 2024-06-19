

#version 450

#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in float height;

layout(location = 1) in vec3 xyzPos;

layout(location = 0) out float hei;

layout(location = 1) out vec4 pos;


void main() {

    hei = height;
    pos = vec4(xyzPos.xyz,1.0);
}
