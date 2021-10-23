#version 330 core

in vec3 pix;
out vec4 fragColor;

uniform sampler2D currentFrame;

void main() {
    vec3 color = texture2D(currentFrame, pix.xy*0.5+0.5).rgb;
    color = pow(color, vec3(1.0/2.2));
    fragColor = vec4(color, 1.0);
}
