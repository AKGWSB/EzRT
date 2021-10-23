#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>
#include <iostream>
#include <algorithm>

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#define INF 114514.0

using namespace glm;

// ----------------------------------------------------------------------------- //

GLuint program;                 // 着色器程序对象
std::vector<vec3> vertices;     // 顶点坐标
std::vector<GLuint> indices;    // 顶点索引
std::vector<vec3> lines;        // 线段端点坐标
vec3 rotateControl(0, 0, 0);    // 旋转参数
vec3 scaleControl(1, 1, 1);     // 缩放参数

// BVH 树节点
struct BVHNode {
    BVHNode* left = NULL;       // 左右子树索引
    BVHNode* right = NULL;
    int n, index;               // 叶子节点信息               
    vec3 AA, BB;                // 碰撞盒
};

typedef struct Triangle {
    vec3 p1, p2, p3;   // 三点
    vec3 center;       // 中心
    Triangle(vec3 a, vec3 b, vec3 c) {
        p1 = a, p2 = b, p3 = c;
        center = (p1 + p2 + p3) / vec3(3, 3, 3);
    }
} Triangle;
std::vector<Triangle> triangles;

// 按照三角形中心排序 -- 比较函数
bool cmpx(const Triangle& t1, const Triangle& t2) {
    return t1.center.x < t2.center.x;
}
bool cmpy(const Triangle& t1, const Triangle& t2) {
    return t1.center.y < t2.center.y;
}
bool cmpz(const Triangle& t1, const Triangle& t2) {
    return t1.center.z < t2.center.z;
}

// 求交结果
struct HitResult {
    Triangle* triangle = NULL;
    float distance = INF;
};

// 光线
typedef struct Ray {
    vec3 startPoint = vec3(0, 0, 0);    // 起点
    vec3 direction = vec3(0, 0, 0);     // 方向
}Ray;

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

// 读取 obj
void readObj(std::string filepath, std::vector<vec3>& vertices, std::vector<GLuint>& indices) {
    // 打开文件流
    std::ifstream fin(filepath);
    std::string line;
    if (!fin.is_open()) {
        std::cout << "文件 " << filepath << " 打开失败" << std::endl;
        exit(-1);
    }

    // 增量读取
    int offset = vertices.size();

    // 按行读取
    while (std::getline(fin, line)) {
        std::istringstream sin(line);   // 以一行的数据作为 string stream 解析并且读取
        std::string type;
        GLfloat x, y, z;
        int v0, v1, v2;

        // 读取obj文件
        sin >> type;
        if (type == "v") {
            sin >> x >> y >> z;
            vertices.push_back(vec3(x, y, z));
        }
        if (type == "f") {
            sin >> v0 >> v1 >> v2;
            indices.push_back(v0 - 1 + offset);
            indices.push_back(v1 - 1 + offset);
            indices.push_back(v2 - 1 + offset);
        }
    }
}

void addLine(vec3 p1, vec3 p2) {
    lines.push_back(p1);
    lines.push_back(p2);
}

void addBox(BVHNode* root) {
    float x1 = root->AA.x, y1 = root->AA.y, z1 = root->AA.z;
    float x2 = root->BB.x, y2 = root->BB.y, z2 = root->BB.z;
    lines.push_back(vec3(x1, y1, z1)), lines.push_back(vec3(x2, y1, z1));
    lines.push_back(vec3(x1, y1, z1)), lines.push_back(vec3(x1, y1, z2));
    lines.push_back(vec3(x1, y1, z1)), lines.push_back(vec3(x1, y2, z1));
    lines.push_back(vec3(x2, y1, z1)), lines.push_back(vec3(x2, y1, z2));
    lines.push_back(vec3(x2, y1, z1)), lines.push_back(vec3(x2, y2, z1));
    lines.push_back(vec3(x1, y2, z1)), lines.push_back(vec3(x2, y2, z1));
    lines.push_back(vec3(x1, y1, z2)), lines.push_back(vec3(x1, y2, z2));
    lines.push_back(vec3(x1, y2, z1)), lines.push_back(vec3(x1, y2, z2));
    lines.push_back(vec3(x1, y2, z2)), lines.push_back(vec3(x2, y2, z2));
    lines.push_back(vec3(x1, y1, z2)), lines.push_back(vec3(x2, y1, z2));
    lines.push_back(vec3(x2, y2, z1)), lines.push_back(vec3(x2, y2, z2));
    lines.push_back(vec3(x2, y1, z2)), lines.push_back(vec3(x2, y2, z2));
}

void addTriangle(Triangle* tri) {
    if (tri) {
        lines.push_back(tri->p1 - vec3(0.0005, 0.0005, 0.0005));
        lines.push_back(tri->p2 - vec3(0.0005, 0.0005, 0.0005));
        lines.push_back(tri->p2 - vec3(0.0005, 0.0005, 0.0005));
        lines.push_back(tri->p3 - vec3(0.0005, 0.0005, 0.0005));
        lines.push_back(tri->p3 - vec3(0.0005, 0.0005, 0.0005));
        lines.push_back(tri->p1 - vec3(0.0005, 0.0005, 0.0005));
        lines.push_back(tri->p1 + vec3(0.0005, 0.0005, 0.0005));
        lines.push_back(tri->p2 + vec3(0.0005, 0.0005, 0.0005));
        lines.push_back(tri->p2 + vec3(0.0005, 0.0005, 0.0005));
        lines.push_back(tri->p3 + vec3(0.0005, 0.0005, 0.0005));
        lines.push_back(tri->p3 + vec3(0.0005, 0.0005, 0.0005));
        lines.push_back(tri->p1 + vec3(0.0005, 0.0005, 0.0005));
    }
}

// 光线和三角形求交 
float hitTriangle(Triangle* triangle, Ray ray) {

    vec3 p1 = triangle->p1, p2 = triangle->p2, p3 = triangle->p3;
    vec3 S = ray.startPoint;        // 射线起点
    vec3 d = ray.direction;         // 射线方向
    vec3 N = normalize(cross(p2 - p1, p3 - p1));    // 法向量
    if (dot(N, d) > 0.0f) N = -N;   // 获取正确的法向量

    // 如果视线和三角形平行
    if (fabs(dot(N, d)) < 0.00001f) return INF;

    // 距离
    float t = (dot(N, p1) - dot(S, N)) / dot(d, N);
    if (t < 0.0005f) return INF;    // 如果三角形在光线背面

    // 交点计算
    vec3 P = S + d * t;

    // 判断交点是否在三角形中
    vec3 c1 = cross(p2 - p1, P - p1);
    vec3 c2 = cross(p3 - p2, P - p2);
    vec3 c3 = cross(p1 - p3, P - p3);
    if (dot(c1, N) > 0 && dot(c2, N) > 0 && dot(c3, N) > 0) return t;
    if (dot(c1, N) < 0 && dot(c2, N) < 0 && dot(c3, N) < 0) return t;

    return INF;
}


// 构建 BVH
BVHNode* buildBVH(std::vector<Triangle>& triangles, int l, int r, int n) {
    if (l > r) return 0;

    BVHNode* node = new BVHNode();
    node->AA = vec3(1145141919, 1145141919, 1145141919);
    node->BB = vec3(-1145141919, -1145141919, -1145141919);

    // 计算 AABB
    for (int i = l; i <= r; i++) {
        // 最小点 AA
        float minx = min(triangles[i].p1.x, min(triangles[i].p2.x, triangles[i].p3.x));
        float miny = min(triangles[i].p1.y, min(triangles[i].p2.y, triangles[i].p3.y));
        float minz = min(triangles[i].p1.z, min(triangles[i].p2.z, triangles[i].p3.z));
        node->AA.x = min(node->AA.x, minx);
        node->AA.y = min(node->AA.y, miny);
        node->AA.z = min(node->AA.z, minz);
        // 最大点 BB
        float maxx = max(triangles[i].p1.x, max(triangles[i].p2.x, triangles[i].p3.x));
        float maxy = max(triangles[i].p1.y, max(triangles[i].p2.y, triangles[i].p3.y));
        float maxz = max(triangles[i].p1.z, max(triangles[i].p2.z, triangles[i].p3.z));
        node->BB.x = max(node->BB.x, maxx);
        node->BB.y = max(node->BB.y, maxy);
        node->BB.z = max(node->BB.z, maxz);
    }

    // 不多于 n 个三角形 返回叶子节点
    if ((r - l + 1) <= n) {
        node->n = r - l + 1;
        node->index = l;
        return node;
    }

    // 否则递归建树
    float lenx = node->BB.x - node->AA.x;
    float leny = node->BB.y - node->AA.y;
    float lenz = node->BB.z - node->AA.z;
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
    node->left = buildBVH(triangles, l, mid, n);
    node->right = buildBVH(triangles, mid + 1, r, n);

    return node;
}

// SAH 优化构建 BVH
BVHNode* buildBVHwithSAH(std::vector<Triangle>& triangles, int l, int r, int n) {
    if (l > r) return 0;

    BVHNode* node = new BVHNode();
    node->AA = vec3(1145141919, 1145141919, 1145141919);
    node->BB = vec3(-1145141919, -1145141919, -1145141919);

    // 计算 AABB
    for (int i = l; i <= r; i++) {
        // 最小点 AA
        float minx = min(triangles[i].p1.x, min(triangles[i].p2.x, triangles[i].p3.x));
        float miny = min(triangles[i].p1.y, min(triangles[i].p2.y, triangles[i].p3.y));
        float minz = min(triangles[i].p1.z, min(triangles[i].p2.z, triangles[i].p3.z));
        node->AA.x = min(node->AA.x, minx);
        node->AA.y = min(node->AA.y, miny);
        node->AA.z = min(node->AA.z, minz);
        // 最大点 BB
        float maxx = max(triangles[i].p1.x, max(triangles[i].p2.x, triangles[i].p3.x));
        float maxy = max(triangles[i].p1.y, max(triangles[i].p2.y, triangles[i].p3.y));
        float maxz = max(triangles[i].p1.z, max(triangles[i].p2.z, triangles[i].p3.z));
        node->BB.x = max(node->BB.x, maxx);
        node->BB.y = max(node->BB.y, maxy);
        node->BB.z = max(node->BB.z, maxz);
    }

    // 不多于 n 个三角形 返回叶子节点
    if ((r - l + 1) <= n) {
        node->n = r - l + 1;
        node->index = l;
        return node;
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
    node->left = buildBVHwithSAH(triangles, l, Split, n);
    node->right = buildBVHwithSAH(triangles, Split + 1, r, n);

    return node;
}

void dfsNlevel(BVHNode* root, int depth, int targetDepth) {
    if (root == NULL) return;
    if (targetDepth == depth) {
        addBox(root);
        return;
    }
    dfsNlevel(root->left, depth + 1, targetDepth);
    dfsNlevel(root->right, depth + 1, targetDepth);
}

// 暴力查数组
HitResult hitTriangleArray(Ray ray, std::vector<Triangle>& triangles, int l, int r) {
    HitResult res;
    for (int i = l; i <= r; i++) {
        float d = hitTriangle(&triangles[i], ray);
        if (d < INF && d < res.distance) {
            res.distance = d;
            res.triangle = &triangles[i];
        }
    }
    return res;
}

// 和 aabb 盒子求交，没有交点则返回 -1
float hitAABB(Ray r, vec3 AA, vec3 BB) {
    // 1.0 / direction
    vec3 invdir = vec3(1.0 / r.direction.x, 1.0 / r.direction.y, 1.0 / r.direction.z);

    vec3 in = (BB - r.startPoint) * invdir;
    vec3 out = (AA - r.startPoint) * invdir;

    vec3 tmax = max(in, out);
    vec3 tmin = min(in, out);

    float t1 = min(tmax.x, min(tmax.y, tmax.z));
    float t0 = max(tmin.x, max(tmin.y, tmin.z));

    return (t1 >= t0) ? ((t0 > 0.0) ? (t0) : (t1)) : (-1);
}

// 在 BVH 上遍历求交
HitResult hitBVH(Ray ray, std::vector<Triangle>& triangles, BVHNode* root) {
    if (root == NULL) return HitResult();

    // 是叶子 暴力查
    if (root->n > 0) {
        return hitTriangleArray(ray, triangles, root->n, root->n + root->index - 1);
    }

    // 和左右子树 AABB 求交
    float d1 = INF, d2 = INF;
    if (root->left) d1 = hitAABB(ray, root->left->AA, root->left->BB);
    if (root->right) d2 = hitAABB(ray, root->right->AA, root->right->BB);

    // 递归结果
    HitResult r1, r2;
    if (d1 > 0) r1 = hitBVH(ray, triangles, root->left);
    if (d2 > 0) r2 = hitBVH(ray, triangles, root->right);

    return r1.distance < r2.distance ? r1 : r2;
}

// ----------------------------------------------------------------------------- //

// 显示回调函数
void display() {

    // 构造模型变换矩阵
    mat4 unit(    // 单位矩阵
        vec4(1, 0, 0, 0),
        vec4(0, 1, 0, 0),
        vec4(0, 0, 1, 0),
        vec4(0, 0, 0, 1)
    );
    mat4 scaleMat = scale(unit, scaleControl);   // xyz缩放0.6倍
    mat4 rotateMat = unit;    // 旋转
    rotateMat = rotate(rotateMat, radians(rotateControl.x), vec3(1, 0, 0)); // 绕x轴转
    rotateMat = rotate(rotateMat, radians(rotateControl.y), vec3(0, 1, 0)); // 绕y轴转
    rotateMat = rotate(rotateMat, radians(rotateControl.z), vec3(0, 0, 1)); // 绕z轴转
    mat4 modelMat = rotateMat * scaleMat;   // 变换级联 -- 生成模型变换矩阵

    GLuint mlocation = glGetUniformLocation(program, "model");  // 名为model的uniform变量的位置索引
    glUniformMatrix4fv(mlocation, 1, GL_FALSE, value_ptr(modelMat));

    GLuint clocation = glGetUniformLocation(program, "color");

    // 绘制
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);       // 清空窗口颜色缓存
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glUniform3fv(clocation, 1, value_ptr(vec3(1, 0, 0)));
    glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);

    // 绘制 AABB 盒子  
    glUniform3fv(clocation, 1, value_ptr(vec3(1, 1, 1)));
    glDrawArrays(GL_LINES, vertices.size(), lines.size());
    glutSwapBuffers();  // 交换缓冲区
}

// 鼠标运动函数
double lastX = 0.0, lastY = 0.0;
void mouse(int x, int y)
{
    // 调整旋转
    rotateControl.y += -200 * (x - lastX) / 512;
    rotateControl.x += -200 * (y - lastY) / 512;
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
    scaleControl.x += 1 * direction * 0.1;
    scaleControl.y += 1 * direction * 0.1;
    scaleControl.z += 1 * direction * 0.1;
    glutPostRedisplay();    // 重绘
}

int main(int argc, char** argv) {

    glutInit(&argc, argv);              // glut初始化
    glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH);
    glutInitWindowSize(512, 512);// 窗口大小
    glutCreateWindow("BVH"); // 创建OpenGL上下文
    glewInit();

    // ------------------------------------------------------------------------------------------- //

    // 读 obj
    readObj("./models/Stanford Bunny.obj", vertices, indices);
    for (auto& v : vertices) {
        v.x *= 5.0, v.y *= 5.0, v.z *= 5.0;
        v.y -= 0.5;
    }
    readObj("./models/quad.obj", vertices, indices);
    // 构建 Triangle 数组
    for (int i = 0; i < indices.size(); i += 3) {
        triangles.push_back(Triangle(vertices[indices[i]], vertices[indices[i + 1]], vertices[indices[i + 2]]));
    }

    // 建立 BVH 树
    //BVHNode* root = buildBVH(triangles, 0, triangles.size() - 1, 8);
    BVHNode* root = buildBVHwithSAH(triangles, 0, triangles.size() - 1, 8);

    //addBox(root->left); 
    //addBox(root->right);

    dfsNlevel(root, 0, 5);   // 可视化第 n 层 bvh

    /**/
    Ray ray;
    ray.startPoint = vec3(0, 0, 1);
    ray.direction = normalize(vec3(0.1, -0.1, -0.7));

    // HitResult res = hitTriangleArray(ray, triangles, 0, triangles.size()-1); // 暴力验证
    HitResult res = hitBVH(ray, triangles, root);
    addTriangle(res.triangle);
    addLine(ray.startPoint, ray.startPoint + ray.direction * vec3(5, 5, 5));


    // ------------------------------------------------------------------------------------------- //

    // 生成vbo对象并且绑定vbo
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vec3) * (vertices.size() + lines.size()), NULL, GL_STATIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vec3) * vertices.size(), vertices.data());
    glBufferSubData(GL_ARRAY_BUFFER, sizeof(vec3) * vertices.size(), sizeof(vec3) * lines.size(), lines.data());

    // 生成vao对象并且绑定vao
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    // 传索引到 ebo
    GLuint ebo;
    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint), indices.data(), GL_STATIC_DRAW);

    // 生成着色器程序对象
    std::string fshaderPath = "./shaders/fshader.fsh";
    std::string vshaderPath = "./shaders/vshader.vsh";
    program = getShaderProgram(fshaderPath, vshaderPath);
    glUseProgram(program);  // 使用着色器

    // 建立顶点变量vPosition在着色器中的索引 同时指定vPosition变量的数据解析格式
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (GLvoid*)0);  // vao指定vPosition变量的数据解析格式

    glEnable(GL_DEPTH_TEST);  // 开启深度测试
    glClearColor(0.0, 0.0, 0.0, 1.0);   // 背景颜色 -- 黑

    glutDisplayFunc(display);   // 设置显示回调函数 -- 每帧执行
    glutMotionFunc(mouse);      // 鼠标拖动
    glutMouseFunc(mouseDown);   // 鼠标左键按下
    glutMouseWheelFunc(mouseWheel); // 滚轮缩放
    glutMainLoop();

    return 0;
}
