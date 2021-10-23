#version 330 core

in vec3 pix;
out vec4 fragColor;

uniform sampler2D currentFrame;

void main() {
    gl_FragData[0] = vec4(vec3(1, 0, 0), 1.0);
    gl_FragData[1] = vec4(vec3(0, 1, 0), 1.0);
    gl_FragData[2] = vec4(vec3(0, 0, 1), 1.0);
}
        
