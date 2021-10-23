#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <ctime>

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>


#include "lib/hdrloader.h"

#define INF 114514.0

using namespace glm;

// ----------------------------------------------------------------------------- //

// 物体表面材质定义
struct Material {
    vec3 emissive = vec3(0, 0, 0);  // 作为光源时的发光颜色
    vec3 baseColor = vec3(1, 1, 1);
    float subsurface = 0.0;
    float metallic = 0.0;
    float specular = 0.5;
    float specularTint = 0.0;
    float roughness = 0.5;
    float anisotropic = 0.0;
    float sheen = 0.0;
    float sheenTint = 0.5;
    float clearcoat = 0.0;
    float clearcoatGloss = 1.0;
    float IOR = 1.0;
    float transmission = 0.0;
};

// 三角形定义
struct Triangle {
    vec3 p1, p2, p3;    // 顶点坐标
    vec3 n1, n2, n3;    // 顶点法线
    Material material;  // 材质
};

// BVH 树节点
struct BVHNode {
    int left, right;    // 左右子树索引
    int n, index;       // 叶子节点信息               
    vec3 AA, BB;        // 碰撞盒
};

// ----------------------------------------------------------------------------- //

struct Triangle_encoded {
    vec3 p1, p2, p3;    // 顶点坐标
    vec3 n1, n2, n3;    // 顶点法线
    vec3 emissive;      // 自发光参数
    vec3 baseColor;     // 颜色
    vec3 param1;        // (subsurface, metallic, specular)
    vec3 param2;        // (specularTint, roughness, anisotropic)
    vec3 param3;        // (sheen, sheenTint, clearcoat)
    vec3 param4;        // (clearcoatGloss, IOR, transmission)
};

struct BVHNode_encoded {
    vec3 childs;        // (left, right, 保留)
    vec3 leafInfo;      // (n, index, 保留)
    vec3 AA, BB;        
};

// ----------------------------------------------------------------------------- //

class RenderPass {
public:
    GLuint FBO = 0;
    GLuint vao, vbo;
    std::vector<GLuint> colorAttachments;
    GLuint program;
    int width = 512;
    int height = 512;
    void bindData(bool finalPass = false) {
        if (!finalPass) glGenFramebuffers(1, &FBO);
        glBindFramebuffer(GL_FRAMEBUFFER, FBO);

        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        std::vector<vec3> square = { vec3(-1, -1, 0), vec3(1, -1, 0), vec3(-1, 1, 0), vec3(1, 1, 0), vec3(-1, 1, 0), vec3(1, -1, 0) };
        glBufferData(GL_ARRAY_BUFFER, sizeof(vec3) * square.size(), NULL, GL_STATIC_DRAW);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vec3) * square.size(), &square[0]);

        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glEnableVertexAttribArray(0);   // layout (location = 0) 
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (GLvoid*)0);
        // 不是 finalPass 则生成帧缓冲的颜色附件
        if (!finalPass) {
            std::vector<GLuint> attachments;
            for (int i = 0; i < colorAttachments.size(); i++) {
                glBindTexture(GL_TEXTURE_2D, colorAttachments[i]);
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, colorAttachments[i], 0);// 将颜色纹理绑定到 i 号颜色附件
                attachments.push_back(GL_COLOR_ATTACHMENT0 + i);
            }
            glDrawBuffers(attachments.size(), &attachments[0]);
        }
        
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
    void draw(std::vector<GLuint> texPassArray = {}) {
        glUseProgram(program);
        glBindFramebuffer(GL_FRAMEBUFFER, FBO);
        glBindVertexArray(vao);
        // 传上一帧的帧缓冲颜色附件
        for (int i = 0; i < texPassArray.size(); i++) {
            glActiveTexture(GL_TEXTURE0+i);
            glBindTexture(GL_TEXTURE_2D, texPassArray[i]);
            std::string uName = "texPass" + std::to_string(i);
            glUniform1i(glGetUniformLocation(program, uName.c_str()), i);
        }
        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        glBindVertexArray(0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glUseProgram(0);
    }
};

// ----------------------------------------------------------------------------- //

GLuint trianglesTextureBuffer;
GLuint nodesTextureBuffer;
GLuint lastFrame;
GLuint hdrMap;

RenderPass pass1;
RenderPass pass2;
RenderPass pass3;

// 相机参数
float upAngle = 0.0;
float rotatAngle = 0.0;
float r = 4.0;

// ----------------------------------------------------------------------------- //

// 按照三角形中心排序 -- 比较函数
bool cmpx(const Triangle& t1, const Triangle& t2) {
    vec3 center1 = (t1.p1 + t1.p2 + t1.p3) / vec3(3, 3, 3);
    vec3 center2 = (t2.p1 + t2.p2 + t2.p3) / vec3(3, 3, 3);
    return center1.x < center2.x;
}
bool cmpy(const Triangle& t1, const Triangle& t2) {
    vec3 center1 = (t1.p1 + t1.p2 + t1.p3) / vec3(3, 3, 3);
    vec3 center2 = (t2.p1 + t2.p2 + t2.p3) / vec3(3, 3, 3);
    return center1.y < center2.y;
}
bool cmpz(const Triangle& t1, const Triangle& t2) {
    vec3 center1 = (t1.p1 + t1.p2 + t1.p3) / vec3(3, 3, 3);
    vec3 center2 = (t2.p1 + t2.p2 + t2.p3) / vec3(3, 3, 3);
    return center1.z < center2.z;
}

// ----------------------------------------------------------------------------- //

// 读取文件并且返回一个长字符串表示文件内容
std::string readShaderFile(std::string filepath) {
    std::string res, line;
    std::ifstream fin(filepath);
    if (!fin.is_open())
    {
        std::cout << "文件 " << filepath << " 打开失败" << std::endl;
        exit(-1);
    }
    while (std::getline(fin, line))
    {
        res += line + '\n';
    }
    fin.close();
    return res;
}

GLuint getTextureRGB32F(int width, int height) {
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    return tex;
}

// 获取着色器对象
GLuint getShaderProgram(std::string fshader, std::string vshader) {
    // 读取shader源文件
    std::string vSource = readShaderFile(vshader);
    std::string fSource = readShaderFile(fshader);
    const char* vpointer = vSource.c_str();
    const char* fpointer = fSource.c_str();

    // 容错
    GLint success;
    GLchar infoLog[512];

    // 创建并编译顶点着色器
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, (const GLchar**)(&vpointer), NULL);
    glCompileShader(vertexShader);
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);   // 错误检测
    if (!success)
    {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cout << "顶点着色器编译错误\n" << infoLog << std::endl;
        exit(-1);
    }

    // 创建并且编译片段着色器
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, (const GLchar**)(&fpointer), NULL);
    glCompileShader(fragmentShader);
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);   // 错误检测
    if (!success)
    {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cout << "片段着色器编译错误\n" << infoLog << std::endl;
        exit(-1);
    }

    // 链接两个着色器到program对象
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    // 删除着色器对象
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return shaderProgram;
}

// ----------------------------------------------------------------------------- //

// 模型变换矩阵
mat4 getTransformMatrix(vec3 rotateCtrl, vec3 translateCtrl, vec3 scaleCtrl) {
    glm::mat4 unit(    // 单位矩阵
        glm::vec4(1, 0, 0, 0),
        glm::vec4(0, 1, 0, 0),
        glm::vec4(0, 0, 1, 0),
        glm::vec4(0, 0, 0, 1)
    );
    mat4 scale = glm::scale(unit, scaleCtrl);
    mat4 translate = glm::translate(unit, translateCtrl);
    mat4 rotate = unit;
    rotate = glm::rotate(rotate, glm::radians(rotateCtrl.x), glm::vec3(1, 0, 0));
    rotate = glm::rotate(rotate, glm::radians(rotateCtrl.y), glm::vec3(0, 1, 0));
    rotate = glm::rotate(rotate, glm::radians(rotateCtrl.z), glm::vec3(0, 0, 1));

    mat4 model = translate * rotate * scale;
    return model;
}

// 读取 obj
void readObj(std::string filepath, std::vector<Triangle>& triangles, Material material, mat4 trans, bool smoothNormal) {

    // 顶点位置，索引
    std::vector<vec3> vertices;
    std::vector<GLuint> indices;

    // 打开文件流
    std::ifstream fin(filepath);
    std::string line;
    if (!fin.is_open()) {
        std::cout << "文件 " << filepath << " 打开失败" << std::endl;
        exit(-1);
    }

    // 计算 AABB 盒，归一化模型大小
    float maxx = -11451419.19;
    float maxy = -11451419.19;
    float maxz = -11451419.19;
    float minx = 11451419.19;
    float miny = 11451419.19;
    float minz = 11451419.19;

    // 按行读取
    while (std::getline(fin, line)) {
        std::istringstream sin(line);   // 以一行的数据作为 string stream 解析并且读取
        std::string type;
        GLfloat x, y, z;
        int v0, v1, v2;
        int vn0, vn1, vn2;
        int vt0, vt1, vt2;
        char slash;

        // 统计斜杆数目，用不同格式读取
        int slashCnt = 0;
        for (int i = 0; i < line.length(); i++) {
            if (line[i] == '/') slashCnt++;
        }

        // 读取obj文件
        sin >> type;
        if (type == "v") {
            sin >> x >> y >> z;
            vertices.push_back(vec3(x, y, z));
            maxx = max(maxx, x); maxy = max(maxx, y); maxz = max(maxx, z);
            minx = min(minx, x); miny = min(minx, y); minz = min(minx, z);
        }
        if (type == "f") {
            if (slashCnt == 6) {
                sin >> v0 >> slash >> vt0 >> slash >> vn0;
                sin >> v1 >> slash >> vt1 >> slash >> vn1;
                sin >> v2 >> slash >> vt2 >> slash >> vn2;
            }
            else if (slashCnt == 3) {
                sin >> v0 >> slash >> vt0;
                sin >> v1 >> slash >> vt1;
                sin >> v2 >> slash >> vt2;
            }
            else {
                sin >> v0 >> v1 >> v2;
            }
            indices.push_back(v0 - 1);
            indices.push_back(v1 - 1);
            indices.push_back(v2 - 1);
        }
    }

    // 模型大小归一化
    float lenx = maxx - minx;
    float leny = maxy - miny;
    float lenz = maxz - minz;
    float maxaxis = max(lenx, max(leny, lenz));
    for (auto& v : vertices) {
        v.x /= maxaxis;
        v.y /= maxaxis;
        v.z /= maxaxis;
    }

    // 通过矩阵进行坐标变换
    for (auto& v : vertices) {
        vec4 vv = vec4(v.x, v.y, v.z, 1);
        vv = trans * vv;
        v = vec3(vv.x, vv.y, vv.z);
    }

    // 生成法线
    std::vector<vec3> normals(vertices.size(), vec3(0, 0, 0));
    for (int i = 0; i < indices.size(); i += 3) {
        vec3 p1 = vertices[indices[i]];
        vec3 p2 = vertices[indices[i + 1]];
        vec3 p3 = vertices[indices[i + 2]];
        vec3 n = normalize(cross(p2 - p1, p3 - p1));
        normals[indices[i]] += n;
        normals[indices[i + 1]] += n;
        normals[indices[i + 2]] += n;
    }

    // 构建 Triangle 对象数组
    int offset = triangles.size();  // 增量更新
    triangles.resize(offset + indices.size() / 3);
    for (int i = 0; i < indices.size(); i += 3) {
        Triangle& t = triangles[offset + i / 3];
        // 传顶点属性
        t.p1 = vertices[indices[i]];
        t.p2 = vertices[indices[i + 1]];
        t.p3 = vertices[indices[i + 2]];
        if (!smoothNormal) {
            vec3 n = normalize(cross(t.p2 - t.p1, t.p3 - t.p1));
            t.n1 = n; t.n2 = n; t.n3 = n;
        }
        else {
            t.n1 = normalize(normals[indices[i]]);
            t.n2 = normalize(normals[indices[i + 1]]);
            t.n3 = normalize(normals[indices[i + 2]]);
        }

        // 传材质
        t.material = material;
    }
}

// 构建 BVH
int buildBVH(std::vector<Triangle>& triangles, std::vector<BVHNode>& nodes, int l, int r, int n) {
    if (l > r) return 0;

    // 注：
    // 此处不可通过指针，引用等方式操作，必须用 nodes[id] 来操作
    // 因为 std::vector<> 扩容时会拷贝到更大的内存，那么地址就改变了
    // 而指针，引用均指向原来的内存，所以会发生错误
    nodes.push_back(BVHNode());
    int id = nodes.size() - 1;   // 注意： 先保存索引
    nodes[id].left = nodes[id].right = nodes[id].n = nodes[id].index = 0;
    nodes[id].AA = vec3(1145141919, 1145141919, 1145141919);
    nodes[id].BB = vec3(-1145141919, -1145141919, -1145141919);

    // 计算 AABB
    for (int i = l; i <= r; i++) {
        // 最小点 AA
        float minx = min(triangles[i].p1.x, min(triangles[i].p2.x, triangles[i].p3.x));
        float miny = min(triangles[i].p1.y, min(triangles[i].p2.y, triangles[i].p3.y));
        float minz = min(triangles[i].p1.z, min(triangles[i].p2.z, triangles[i].p3.z));
        nodes[id].AA.x = min(nodes[id].AA.x, minx);
        nodes[id].AA.y = min(nodes[id].AA.y, miny);
        nodes[id].AA.z = min(nodes[id].AA.z, minz);
        // 最大点 BB
        float maxx = max(triangles[i].p1.x, max(triangles[i].p2.x, triangles[i].p3.x));
        float maxy = max(triangles[i].p1.y, max(triangles[i].p2.y, triangles[i].p3.y));
        float maxz = max(triangles[i].p1.z, max(triangles[i].p2.z, triangles[i].p3.z));
        nodes[id].BB.x = max(nodes[id].BB.x, maxx);
        nodes[id].BB.y = max(nodes[id].BB.y, maxy);
        nodes[id].BB.z = max(nodes[id].BB.z, maxz);
    }

    // 不多于 n 个三角形 返回叶子节点
    if ((r - l + 1) <= n) {
        nodes[id].n = r - l + 1;
        nodes[id].index = l;
        return id;
    }

    // 否则递归建树
    float lenx = nodes[id].BB.x - nodes[id].AA.x;
    float leny = nodes[id].BB.y - nodes[id].AA.y;
    float lenz = nodes[id].BB.z - nodes[id].AA.z;
    // 按 x 划分
    if (lenx >= leny && lenx >= lenz)
        std::sort(triangles.begin() + l, triangles.begin() + r + 1, cmpx);
    // 按 y 划分
    if (leny >= lenx && leny >= lenz)
        std::sort(triangles.begin() + l, triangles.begin() + r + 1, cmpy);
    // 按 z 划分
    if (lenz >= lenx && lenz >= leny)
        std::sort(triangles.begin() + l, triangles.begin() + r + 1, cmpz);
    // 递归
    int mid = (l + r) / 2;
    int left = buildBVH(triangles, nodes, l, mid, n);
    int right = buildBVH(triangles, nodes, mid + 1, r, n);

    nodes[id].left = left;
    nodes[id].right = right;

    return id;
}

// SAH 优化构建 BVH
int buildBVHwithSAH(std::vector<Triangle>& triangles, std::vector<BVHNode>& nodes, int l, int r, int n) {
    if (l > r) return 0;

    nodes.push_back(BVHNode());
    int id = nodes.size() - 1;   
    nodes[id].left = nodes[id].right = nodes[id].n = nodes[id].index = 0;
    nodes[id].AA = vec3(1145141919, 1145141919, 1145141919);
    nodes[id].BB = vec3(-1145141919, -1145141919, -1145141919);

    // 计算 AABB
    for (int i = l; i <= r; i++) {
        // 最小点 AA
        float minx = min(triangles[i].p1.x, min(triangles[i].p2.x, triangles[i].p3.x));
        float miny = min(triangles[i].p1.y, min(triangles[i].p2.y, triangles[i].p3.y));
        float minz = min(triangles[i].p1.z, min(triangles[i].p2.z, triangles[i].p3.z));
        nodes[id].AA.x = min(nodes[id].AA.x, minx);
        nodes[id].AA.y = min(nodes[id].AA.y, miny);
        nodes[id].AA.z = min(nodes[id].AA.z, minz);
        // 最大点 BB
        float maxx = max(triangles[i].p1.x, max(triangles[i].p2.x, triangles[i].p3.x));
        float maxy = max(triangles[i].p1.y, max(triangles[i].p2.y, triangles[i].p3.y));
        float maxz = max(triangles[i].p1.z, max(triangles[i].p2.z, triangles[i].p3.z));
        nodes[id].BB.x = max(nodes[id].BB.x, maxx);
        nodes[id].BB.y = max(nodes[id].BB.y, maxy);
        nodes[id].BB.z = max(nodes[id].BB.z, maxz);
    }

    // 不多于 n 个三角形 返回叶子节点
    if ((r - l + 1) <= n) {
        nodes[id].n = r - l + 1;
        nodes[id].index = l;
        return id;
    }

    // 否则递归建树
    float Cost = INF;
    int Axis = 0;
    int Split = (l + r) / 2;
    for (int axis = 0; axis < 3; axis++) {
        // 分别按 x，y，z 轴排序
        if (axis == 0) std::sort(&triangles[0] + l, &triangles[0] + r + 1, cmpx);
        if (axis == 1) std::sort(&triangles[0] + l, &triangles[0] + r + 1, cmpy);
        if (axis == 2) std::sort(&triangles[0] + l, &triangles[0] + r + 1, cmpz);

        // leftMax[i]: [l, i] 中最大的 xyz 值
        // leftMin[i]: [l, i] 中最小的 xyz 值
        std::vector<vec3> leftMax(r - l + 1, vec3(-INF, -INF, -INF));
        std::vector<vec3> leftMin(r - l + 1, vec3(INF, INF, INF));
        // 计算前缀 注意 i-l 以对齐到下标 0
        for (int i = l; i <= r; i++) {
            Triangle& t = triangles[i];
            int bias = (i == l) ? 0 : 1;  // 第一个元素特殊处理

            leftMax[i - l].x = max(leftMax[i - l - bias].x, max(t.p1.x, max(t.p2.x, t.p3.x)));
            leftMax[i - l].y = max(leftMax[i - l - bias].y, max(t.p1.y, max(t.p2.y, t.p3.y)));
            leftMax[i - l].z = max(leftMax[i - l - bias].z, max(t.p1.z, max(t.p2.z, t.p3.z)));

            leftMin[i - l].x = min(leftMin[i - l - bias].x, min(t.p1.x, min(t.p2.x, t.p3.x)));
            leftMin[i - l].y = min(leftMin[i - l - bias].y, min(t.p1.y, min(t.p2.y, t.p3.y)));
            leftMin[i - l].z = min(leftMin[i - l - bias].z, min(t.p1.z, min(t.p2.z, t.p3.z)));
        }

        // rightMax[i]: [i, r] 中最大的 xyz 值
        // rightMin[i]: [i, r] 中最小的 xyz 值
        std::vector<vec3> rightMax(r - l + 1, vec3(-INF, -INF, -INF));
        std::vector<vec3> rightMin(r - l + 1, vec3(INF, INF, INF));
        // 计算后缀 注意 i-l 以对齐到下标 0
        for (int i = r; i >= l; i--) {
            Triangle& t = triangles[i];
            int bias = (i == r) ? 0 : 1;  // 第一个元素特殊处理

            rightMax[i - l].x = max(rightMax[i - l + bias].x, max(t.p1.x, max(t.p2.x, t.p3.x)));
            rightMax[i - l].y = max(rightMax[i - l + bias].y, max(t.p1.y, max(t.p2.y, t.p3.y)));
            rightMax[i - l].z = max(rightMax[i - l + bias].z, max(t.p1.z, max(t.p2.z, t.p3.z)));

            rightMin[i - l].x = min(rightMin[i - l + bias].x, min(t.p1.x, min(t.p2.x, t.p3.x)));
            rightMin[i - l].y = min(rightMin[i - l + bias].y, min(t.p1.y, min(t.p2.y, t.p3.y)));
            rightMin[i - l].z = min(rightMin[i - l + bias].z, min(t.p1.z, min(t.p2.z, t.p3.z)));
        }

        // 遍历寻找分割
        float cost = INF;
        int split = l;
        for (int i = l; i <= r - 1; i++) {
            float lenx, leny, lenz;
            // 左侧 [l, i]
            vec3 leftAA = leftMin[i - l];
            vec3 leftBB = leftMax[i - l];
            lenx = leftBB.x - leftAA.x;
            leny = leftBB.y - leftAA.y;
            lenz = leftBB.z - leftAA.z;
            float leftS = 2.0 * ((lenx * leny) + (lenx * lenz) + (leny * lenz));
            float leftCost = leftS * (i - l + 1);

            // 右侧 [i+1, r]
            vec3 rightAA = rightMin[i + 1 - l];
            vec3 rightBB = rightMax[i + 1 - l];
            lenx = rightBB.x - rightAA.x;
            leny = rightBB.y - rightAA.y;
            lenz = rightBB.z - rightAA.z;
            float rightS = 2.0 * ((lenx * leny) + (lenx * lenz) + (leny * lenz));
            float rightCost = rightS * (r - i);

            // 记录每个分割的最小答案
            float totalCost = leftCost + rightCost;
            if (totalCost < cost) {
                cost = totalCost;
                split = i;
            }
        }
        // 记录每个轴的最佳答案
        if (cost < Cost) {
            Cost = cost;
            Axis = axis;
            Split = split;
        }
    }

    // 按最佳轴分割
    if (Axis == 0) std::sort(&triangles[0] + l, &triangles[0] + r + 1, cmpx);
    if (Axis == 1) std::sort(&triangles[0] + l, &triangles[0] + r + 1, cmpy);
    if (Axis == 2) std::sort(&triangles[0] + l, &triangles[0] + r + 1, cmpz);

    // 递归
    int left  = buildBVHwithSAH(triangles, nodes, l, Split, n);
    int right = buildBVHwithSAH(triangles, nodes, Split + 1, r, n);

    nodes[id].left = left;
    nodes[id].right = right;

    return id;
}

// ----------------------------------------------------------------------------- //

// 绘制
clock_t t1, t2;
double dt, fps;
unsigned int frameCounter = 0;
void display() {
    
    // 帧计时
    t2 = clock();
    dt = (double)(t2 - t1) / CLOCKS_PER_SEC;
    fps = 1.0 / dt;
    std::cout << "\r";
    std::cout << std::fixed << std::setprecision(2) << "FPS : " << fps << "    迭代次数: " << frameCounter;
    t1 = t2;

    // 相机参数
    vec3 eye = vec3(-sin(radians(rotatAngle)) * cos(radians(upAngle)), sin(radians(upAngle)), cos(radians(rotatAngle)) * cos(radians(upAngle)));
    eye.x *= r; eye.y *= r; eye.z *= r;
    mat4 cameraRotate = lookAt(eye, vec3(0, 0, 0), vec3(0, 1, 0));  // 相机注视着原点
    cameraRotate = inverse(cameraRotate);   // lookat 的逆矩阵将光线方向进行转换

    // 传 uniform 给 pass1
    glUseProgram(pass1.program);
    glUniform3fv(glGetUniformLocation(pass1.program, "eye"), 1, value_ptr(eye));
    glUniformMatrix4fv(glGetUniformLocation(pass1.program, "cameraRotate"), 1, GL_FALSE, value_ptr(cameraRotate));
    glUniform1ui(glGetUniformLocation(pass1.program, "frameCounter"), frameCounter++);// 传计数器用作随机种子

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_BUFFER, trianglesTextureBuffer);
    glUniform1i(glGetUniformLocation(pass1.program, "triangles"), 0);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_BUFFER, nodesTextureBuffer);
    glUniform1i(glGetUniformLocation(pass1.program, "nodes"), 1);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, lastFrame);
    glUniform1i(glGetUniformLocation(pass1.program, "lastFrame"), 2);

    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, hdrMap);
    glUniform1i(glGetUniformLocation(pass1.program, "hdrMap"), 3);

    // 绘制
    pass1.draw();
    pass2.draw(pass1.colorAttachments);
    pass3.draw(pass2.colorAttachments);

    glutSwapBuffers();
}

// 每一帧
void frameFunc() {
    glutPostRedisplay();
}

// 鼠标运动函数
double lastX = 0.0, lastY = 0.0;
void mouse(int x, int y) {
    frameCounter = 0;
    // 调整旋转
    rotatAngle += 150 * (x - lastX) / 512;
    upAngle += 150 * (y - lastY) / 512;
    upAngle = min(upAngle, 89.0f);
    upAngle = max(upAngle, -89.0f);
    lastX = x, lastY = y;
    glutPostRedisplay();    // 重绘
}

// 鼠标按下
void mouseDown(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
        lastX = x, lastY = y;
    }
}

// 鼠标滚轮函数
void mouseWheel(int wheel, int direction, int x, int y) {
    frameCounter = 0;
    r += -direction * 0.5;
    glutPostRedisplay();    // 重绘
}

// ----------------------------------------------------------------------------- //

int main(int argc, char** argv) {

    glutInit(&argc, argv);              // glut初始化
    glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH);
    glutInitWindowSize(512, 512);// 窗口大小
    glutInitWindowPosition(350, 50);
    glutCreateWindow("Path Tracing GPU"); // 创建OpenGL上下文
    glewInit();

    // ----------------------------------------------------------------------------- //

    // scene config
    std::vector<Triangle> triangles;

    Material m;
    m.baseColor = vec3(1, 0.5, 0.5);
    m.roughness = 0.1;
    m.metallic = 0.0;
    m.clearcoat = 1.0;
    m.subsurface = 1.0;
    //m.clearcoatGloss = 0.05;
    //readObj("models/sphere2.obj", triangles, m, getTransformMatrix(vec3(0, 0, 0), vec3(-1, -0.85, 0), vec3(0.75, 0.75, 0.75)),true);
    //readObj("models/Stanford Bunny.obj", triangles, m, getTransformMatrix(vec3(0, 0, 0), vec3(-2.6, -1.5, 0), vec3(2.5, 2.5, 2.5)), true);

    m.baseColor = vec3(0.5, 1, 0.5);
    m.baseColor = vec3(0.75, 0.7, 0.15);
    m.roughness = 0.15;
    m.metallic = 1.0;
    m.clearcoat = 1.0;
    //m.emissive = vec3(10, 20, 10);
    //readObj("models/sphere2.obj", triangles, m, getTransformMatrix(vec3(0, 0, 0), vec3(0, -0.85, 0), vec3(0.75, 0.75, 0.75)), true);
    readObj("models/teapot.obj", triangles, m, getTransformMatrix(vec3(0, 0, 0), vec3(0, -0.4, 0), vec3(1.75, 1.75, 1.75)), true);
    //readObj("models/Stanford Bunny.obj", triangles, m, getTransformMatrix(vec3(0, 0, 0), vec3(0.4, -1.5, 0), vec3(2.5, 2.5, 2.5)), true);

    m.baseColor = vec3(0.5, 0.5, 1);
    m.metallic = 0.0;
    m.roughness = 0.1;
    m.clearcoat = 1.0;
    //m.emissive = vec3(10, 10, 20);
    //readObj("models/sphere2.obj", triangles, m, getTransformMatrix(vec3(0, 0, 0), vec3(1, -0.85, 0), vec3(0.75, 0.75, 0.75)), true);
    //readObj("models/Stanford Bunny.obj", triangles, m, getTransformMatrix(vec3(0, 0, 0), vec3(3.4, -1.5, 0), vec3(2.5, 2.5, 2.5)), true);
    
    m.emissive = vec3(0, 0, 0);
    m.baseColor = vec3(0.725, 0.71, 0.68);
    m.baseColor = vec3(1,1,1);
    
    float len = 13.0;
    m.metallic = 0.7;
    m.roughness = 0.3;
    //readObj("models/quad.obj", triangles, m, getTransformMatrix(vec3(0, 0, 0), vec3(0, -1.5, 0), vec3(len, 0.01, len)), false);
    
    m.baseColor = vec3(1, 1, 1);
    m.emissive = vec3(20, 20, 20);
    //readObj("models/quad.obj", triangles, m, getTransformMatrix(vec3(0, 0, 0), vec3(0.0, 1.48, -0.0), vec3(1.7, 0.01, 1.7)), false);
    

    int nTriangles = triangles.size();
    std::cout << "模型读取完成: 共 " << nTriangles << " 个三角形" << std::endl;

    // 建立 bvh
    BVHNode testNode;
    testNode.left = 255;
    testNode.right = 128;
    testNode.n = 30;
    testNode.AA = vec3(1, 1, 0);
    testNode.BB = vec3(0, 1, 0);
    std::vector<BVHNode> nodes{ testNode };
    //buildBVH(triangles, nodes, 0, triangles.size() - 1, 8);
    buildBVHwithSAH(triangles, nodes, 0, triangles.size() - 1, 8);
    int nNodes = nodes.size();
    std::cout << "BVH 建立完成: 共 " << nNodes << " 个节点" << std::endl;

    // 编码 三角形, 材质
    std::vector<Triangle_encoded> triangles_encoded(nTriangles);
    for (int i = 0; i < nTriangles; i++) {
        Triangle& t = triangles[i];
        Material& m = t.material;
        // 顶点位置
        triangles_encoded[i].p1 = t.p1;
        triangles_encoded[i].p2 = t.p2;
        triangles_encoded[i].p3 = t.p3;
        // 顶点法线
        triangles_encoded[i].n1 = t.n1;
        triangles_encoded[i].n2 = t.n2;
        triangles_encoded[i].n3 = t.n3;
        // 材质
        triangles_encoded[i].emissive = m.emissive;
        triangles_encoded[i].baseColor = m.baseColor;
        triangles_encoded[i].param1 = vec3(m.subsurface, m.metallic, m.specular);
        triangles_encoded[i].param2 = vec3(m.specularTint, m.roughness, m.anisotropic);
        triangles_encoded[i].param3 = vec3(m.sheen, m.sheenTint, m.clearcoat);
        triangles_encoded[i].param4 = vec3(m.clearcoatGloss, m.IOR, m.transmission);
    }

    // 编码 BVHNode, aabb
    std::vector<BVHNode_encoded> nodes_encoded(nNodes);
    for (int i = 0; i < nNodes; i++) {
        nodes_encoded[i].childs = vec3(nodes[i].left, nodes[i].right, 0);
        nodes_encoded[i].leafInfo = vec3(nodes[i].n, nodes[i].index, 0);
        nodes_encoded[i].AA = nodes[i].AA;
        nodes_encoded[i].BB = nodes[i].BB;
    }

    // ----------------------------------------------------------------------------- //

    // 生成纹理

    // 三角形数组
    GLuint tbo0;
    glGenBuffers(1, &tbo0);
    glBindBuffer(GL_TEXTURE_BUFFER, tbo0);
    glBufferData(GL_TEXTURE_BUFFER, triangles_encoded.size() * sizeof(Triangle_encoded), &triangles_encoded[0], GL_STATIC_DRAW);
    glGenTextures(1, &trianglesTextureBuffer);
    glBindTexture(GL_TEXTURE_BUFFER, trianglesTextureBuffer);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_RGB32F, tbo0);

    // BVHNode 数组
    GLuint tbo1;
    glGenBuffers(1, &tbo1);
    glBindBuffer(GL_TEXTURE_BUFFER, tbo1);
    glBufferData(GL_TEXTURE_BUFFER, nodes_encoded.size() * sizeof(BVHNode_encoded), &nodes_encoded[0], GL_STATIC_DRAW);
    glGenTextures(1, &nodesTextureBuffer);
    glBindTexture(GL_TEXTURE_BUFFER, nodesTextureBuffer);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_RGB32F, tbo1);

    // hdr 全景图
    HDRLoaderResult hdrRes;
    bool r = HDRLoader::load("./HDR/peppermint_powerplant_4k.hdr", hdrRes);
    hdrMap = getTextureRGB32F(hdrRes.width, hdrRes.height);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, hdrRes.width, hdrRes.height, 0, GL_RGB, GL_FLOAT, hdrRes.cols);

    // ----------------------------------------------------------------------------- //

    // 管线配置

    pass1.program = getShaderProgram("./shaders/fshader.fsh", "./shaders/vshader.vsh");
    //pass1.width = pass1.height = 256;
    pass1.colorAttachments.push_back(getTextureRGB32F(pass1.width, pass1.height));
    pass1.colorAttachments.push_back(getTextureRGB32F(pass1.width, pass1.height));
    pass1.colorAttachments.push_back(getTextureRGB32F(pass1.width, pass1.height));
    pass1.bindData();
   
    glUseProgram(pass1.program);
    glUniform1i(glGetUniformLocation(pass1.program, "nTriangles"), triangles.size());
    glUniform1i(glGetUniformLocation(pass1.program, "nNodes"), nodes.size());
    glUniform1i(glGetUniformLocation(pass1.program, "width"), pass1.width);
    glUniform1i(glGetUniformLocation(pass1.program, "height"), pass1.height);
    glUseProgram(0);

    pass2.program = getShaderProgram("./shaders/pass2.fsh", "./shaders/vshader.vsh");
    lastFrame = getTextureRGB32F(pass2.width, pass2.height);
    pass2.colorAttachments.push_back(lastFrame);
    pass2.bindData();

    pass3.program = getShaderProgram("./shaders/pass3.fsh", "./shaders/vshader.vsh");
    pass3.bindData(true);


    // ----------------------------------------------------------------------------- //

    std::cout << "开始..." << std::endl << std::endl;

    glEnable(GL_DEPTH_TEST);  // 开启深度测试
    glClearColor(0.0, 0.0, 0.0, 1.0);   // 背景颜色 -- 黑

    glutDisplayFunc(display);   // 设置显示回调函数
    glutIdleFunc(frameFunc);    // 闲置时刷新
    glutMotionFunc(mouse);      // 鼠标拖动
    glutMouseFunc(mouseDown);   // 鼠标左键按下
    glutMouseWheelFunc(mouseWheel); // 滚轮缩放
    glutMainLoop();

    return 0;
}
