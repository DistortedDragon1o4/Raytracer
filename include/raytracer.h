#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <nlohmann/json.hpp>

#include "stb/stb_image_write.h"

using json = nlohmann::json;

#include "shaderCompiler.h"
#include "VAO.h"
#include "buffers.h"
#include "texture.h"

struct Vertex {
	float x;
	float y;
	float z;
	float pad1;

	float nx;
	float ny;
	float nz;
	int id;
};

struct Triangle {
	Vertex v1;
	Vertex v2;
	Vertex v3;
};

struct Light {
	float x;
	float y;
	float z;
	float rad;

	float r;
	float g;
	float b;
	float intensity;
};

struct Material {
	glm::vec3 color;
    float subsurface;
    glm::vec3 sheen_tint;
    float sheen;
    glm::vec3 specular_tint;
    float specular_strength;
    float roughness;
    float metallic;
    float pad1;
    float pad2;
};

struct RayTraceManager {
	newVAO vertexArray;

	UnifiedGLBufferContainer vertexBuffer;
	UnifiedGLBufferContainer lightBuffer;
	UnifiedGLBufferContainer materialBuffer;
	Texture texture;

	Compute raytracer;
	Shader renderShader;

	int width;
	int height;

	std::vector<unsigned char> pixels;

	std::vector<Triangle> mesh;
	std::vector<Material> materials;


	std::vector<Light> lights;

	RayTraceManager(int width, int height);

	std::vector<Triangle> parseOBJFile(const std::string& filePath, int id);
	Material parseMaterialFromJSON(const std::string& filepath);
	Light parseLightFromJSON(const std::string& filepath);

	void loadGeometry();
	void loadLights();

	void executeComputeOperation();
	void executeRenderOperation();
};
