#version 330 core

layout (location = 0) in vec3 vPosition;  // cpu传入的顶点坐标

uniform mat4 model;

void main() {
    gl_Position = model * vec4(vPosition, 1.0);
}

