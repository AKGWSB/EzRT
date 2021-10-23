#version 330 core

out vec4 fColor;    // 片元输出像素的颜色

uniform vec3 color;

void main() {
    fColor.rgb = color;
}
