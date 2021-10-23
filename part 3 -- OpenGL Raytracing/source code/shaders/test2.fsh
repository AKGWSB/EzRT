#version 330 core

in vec3 pix;
out vec4 fragColor;

uniform sampler2D texPass;
uniform sampler2D texPass0;
uniform sampler2D texPass1;
uniform sampler2D texPass2;
uniform sampler2D texPass3;
uniform sampler2D texPass4;
uniform sampler2D texPass5;
uniform sampler2D texPass6;

void main() {
    vec3 color;
    
    if(pix.x>0 && pix.y>0)
        color = texture2D(texPass0, pix.xy).rgb;
    
    fragColor = vec4(color, 1.0);
}
