#include "glad/gl.h"
#include "glm/ext/matrix_clip_space.hpp"
#include "glm/ext/matrix_transform.hpp"
#include <raytracer.h>
#include <cstring>
#include <vector>

RayTraceManager::RayTraceManager(int width, int height) :
	texture(GL_TEXTURE_2D, 0, width, height),
	renderShader("shaders/vertex.glsl", "shaders/fragment.glsl"),
	raytracer("shaders/raytracer.glsl") {

		this->width = width;
		this->height = height;

		pixels = std::vector<unsigned char>(width * height * 4);

		vertexArray.create();
		vertexArray.bind();
}

Material RayTraceManager::parseMaterialFromJSON(const std::string& filepath) {
    Material material;

    // Open the JSON file
    std::ifstream inputFile(filepath);
    if (!inputFile.is_open()) {
        throw std::runtime_error("Failed to open JSON file: " + filepath);
    }

    // Parse the JSON file
    json jsonData;
    inputFile >> jsonData;

    // Extract data from JSON and populate the Material struct
    material.color = glm::vec3(
        jsonData["color"][0], jsonData["color"][1], jsonData["color"][2]
    );
    material.roughness = jsonData["roughness"];
    material.subsurface = jsonData["subsurface"];
    material.sheen = jsonData["sheen"];
    material.sheen_tint = glm::vec3(
        jsonData["sheen_tint"][0], jsonData["sheen_tint"][1], jsonData["sheen_tint"][2]
    );
    material.specular_strength = jsonData["specular_strength"];
    material.specular_tint = glm::vec3(
        jsonData["specular_tint"][0], jsonData["specular_tint"][1], jsonData["specular_tint"][2]
    );
    material.metallic = jsonData["metallic"];

    return material;
}

std::vector<Triangle> RayTraceManager::parseOBJFile(const std::string& filePath, int id) {
    std::vector<Triangle> triangles;
    std::vector<glm::vec3> tempPositions; // Temporary storage for vertex positions
    std::vector<glm::vec3> tempNormals;   // Temporary storage for vertex normals

    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filePath << std::endl;
        return triangles;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string type;
        iss >> type;

        // Parse vertex positions (v x y z)
        if (type == "v") {
            glm::vec3 position;
            iss >> position.x >> position.y >> position.z;
            tempPositions.push_back(position);
        }

        // Parse vertex normals (vn x y z)
        else if (type == "vn") {
            glm::vec3 normal;
            iss >> normal.x >> normal.y >> normal.z;
            tempNormals.push_back(normal);
        }

        // Parse faces (f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3)
        else if (type == "f") {
            std::string v1, v2, v3;
            iss >> v1 >> v2 >> v3;

            // Extract vertex indices and normal indices
            auto extractIndices = [](const std::string& vertexData, int& posIndex, int& normalIndex) {
                size_t slash1 = vertexData.find('/');
                size_t slash2 = vertexData.find('/', slash1 + 1);
                posIndex = std::stoi(vertexData.substr(0, slash1)) - 1; // OBJ indices start at 1
                normalIndex = std::stoi(vertexData.substr(slash2 + 1)) - 1;
            };

            int posIndex1, normalIndex1;
            int posIndex2, normalIndex2;
            int posIndex3, normalIndex3;
            extractIndices(v1, posIndex1, normalIndex1);
            extractIndices(v2, posIndex2, normalIndex2);
            extractIndices(v3, posIndex3, normalIndex3);

            // Create vertices for the triangle
            Vertex vertex1 = {
                tempPositions[posIndex1].x, tempPositions[posIndex1].y, tempPositions[posIndex1].z, 0.0,
                tempNormals[normalIndex1].x, tempNormals[normalIndex1].y, tempNormals[normalIndex1].z, id,
            };
            Vertex vertex2 = {
                tempPositions[posIndex2].x, tempPositions[posIndex2].y, tempPositions[posIndex2].z, 0.0,
                tempNormals[normalIndex2].x, tempNormals[normalIndex2].y, tempNormals[normalIndex2].z, id,
            };
            Vertex vertex3 = {
                tempPositions[posIndex3].x, tempPositions[posIndex3].y, tempPositions[posIndex3].z, 0.0,
                tempNormals[normalIndex3].x, tempNormals[normalIndex3].y, tempNormals[normalIndex3].z, id,
            };

            // Create a Triangle object and add it to the vector
            Triangle triangle = {vertex1, vertex2, vertex3};
            triangles.push_back(triangle);
        }
    }

    file.close();
    return triangles;
}

Light RayTraceManager::parseLightFromJSON(const std::string& filepath) {
    Light light;

    // Open the JSON file
    std::ifstream inputFile(filepath);
    if (!inputFile.is_open()) {
        throw std::runtime_error("Failed to open JSON file: " + filepath);
    }

    // Parse the JSON file
    json jsonData;
    inputFile >> jsonData;

    // Extract data from JSON and populate the Light struct
    light.x = jsonData["x"];
    light.y = jsonData["y"];
    light.z = jsonData["z"];
    light.rad = jsonData["rad"];

    light.r = jsonData["r"];
    light.g = jsonData["g"];
    light.b = jsonData["b"];
    light.intensity = jsonData["intensity"];

    return light;
}

void flipImageVertically(int width, int height, std::vector<unsigned char>& pixels) {
    int rowSize = width * 4; // 4 bytes per pixel (RGBA)
    std::vector<unsigned char> rowBuffer(rowSize);

    for (int y = 0; y < height / 2; ++y) {
        // Get pointers to the current and corresponding row from the bottom
        unsigned char* row1 = pixels.data() + y * rowSize;
        unsigned char* row2 = pixels.data() + (height - 1 - y) * rowSize;

        // Swap the rows
        std::memcpy(rowBuffer.data(), row1, rowSize);
        std::memcpy(row1, row2, rowSize);
        std::memcpy(row2, rowBuffer.data(), rowSize);
    }
}

void printTriangles(const std::vector<Triangle>& triangles) {
    for (const auto& triangle : triangles) {
        std::cout << "Triangle:\n";
        std::cout << "  v1: (" << triangle.v1.x << ", " << triangle.v1.y << ", " << triangle.v1.z << "), "
                  << "Normal: (" << triangle.v1.nx << ", " << triangle.v1.ny << ", " << triangle.v1.nz << ")\n";
        std::cout << "  v2: (" << triangle.v2.x << ", " << triangle.v2.y << ", " << triangle.v2.z << "), "
                  << "Normal: (" << triangle.v2.nx << ", " << triangle.v2.ny << ", " << triangle.v2.nz << ")\n";
        std::cout << "  v3: (" << triangle.v3.x << ", " << triangle.v3.y << ", " << triangle.v3.z << "), "
                  << "Normal: (" << triangle.v3.nx << ", " << triangle.v3.ny << ", " << triangle.v3.nz << ")\n";
    }
}

void RayTraceManager::loadGeometry() {

	for (int i = 0; i < 4; i++) {
		std::vector<Triangle> crntMesh = parseOBJFile("assets/model" + std::to_string(i) + ".obj", i);
		mesh.insert(mesh.end(), crntMesh.begin(), crntMesh.end());
		materials.push_back(parseMaterialFromJSON("assets/model" + std::to_string(i) + ".json"));
	}

	vertexBuffer.create();
	vertexBuffer.allocate(mesh.size() * sizeof(Triangle), GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT);
	vertexBuffer.map(0, mesh.size() * sizeof(Triangle), GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_FLUSH_EXPLICIT_BIT);

	std::memcpy(vertexBuffer.persistentMappedPtr, mesh.data(), mesh.size() * sizeof(Triangle));
	glFlushMappedNamedBufferRange(vertexBuffer.ID, 0, mesh.size() * sizeof(Triangle));

	vertexBuffer.bindBufferBase(GL_SHADER_STORAGE_BUFFER, 0);
	vertexBuffer.bind(GL_SHADER_STORAGE_BUFFER);

	materialBuffer.create();
	materialBuffer.allocate(materials.size() * sizeof(Material), GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT);
	materialBuffer.map(0, materials.size() * sizeof(Material), GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_FLUSH_EXPLICIT_BIT);

	std::memcpy(materialBuffer.persistentMappedPtr, materials.data(), materials.size() * sizeof(Material));
	glFlushMappedNamedBufferRange(materialBuffer.ID, 0, materials.size() * sizeof(Material));

	materialBuffer.bindBufferBase(GL_SHADER_STORAGE_BUFFER, 2);
	materialBuffer.bind(GL_SHADER_STORAGE_BUFFER);
}

void RayTraceManager::loadLights() {
	for (int i = 0; i < 1; i++) {
		lights.push_back(parseLightFromJSON("assets/light" + std::to_string(i) + ".json"));
	}
	// lights = {{6.07625, 0.90386, 6.00545, 2.5, 1.0, 1.0, 1.0, 150.0}};

	lightBuffer.create();
	lightBuffer.allocate(lights.size() * sizeof(Light), GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT);
	lightBuffer.map(0, lights.size() * sizeof(Light), GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_FLUSH_EXPLICIT_BIT);

	std::memcpy(lightBuffer.persistentMappedPtr, lights.data(), lights.size() * sizeof(Light));
	glFlushMappedNamedBufferRange(vertexBuffer.ID, 0, lights.size() * sizeof(Light));

	lightBuffer.bindBufferBase(GL_SHADER_STORAGE_BUFFER, 1);
	lightBuffer.bind(GL_SHADER_STORAGE_BUFFER);
}

void RayTraceManager::executeComputeOperation() {

	glm::vec3 cameraPosition = glm::vec3(7.3589, 4.9583, 6.9258);
	glm::vec3 cameraLookingAt = glm::vec3(-0.6516, -0.4453, -0.6142);

	glm::mat4 view = glm::lookAt(cameraPosition, cameraLookingAt, glm::vec3(0.0, 1.0, 0.0));
	glm::mat4 projection = glm::perspective(float(glm::radians(55.0)), float(width) / float(height), 0.125f, 1024.0f);

	glm::mat4 cameraMatrix = projection * view;

	texture.BindTexImage();
	raytracer.Activate();

	glUniform1i(glGetUniformLocation(raytracer.ID, "width"), width);
	glUniform1i(glGetUniformLocation(raytracer.ID, "height"), height);

	glUniform1i(glGetUniformLocation(raytracer.ID, "meshSize"), mesh.size());
	glUniform1i(glGetUniformLocation(raytracer.ID, "lightSize"), lights.size());

	glUniform3fv(glGetUniformLocation(raytracer.ID, "cameraPosition"), 1, glm::value_ptr(cameraPosition));

	glUniformMatrix4fv(glGetUniformLocation(raytracer.ID, "matrix"), 1, GL_FALSE, glm::value_ptr(glm::mat4(cameraMatrix)));

	glDispatchCompute(width / 16, height / 16, 1);

	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

	GLsync gpuFence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);

	GLenum waitReturn = GL_UNSIGNALED;
	while (waitReturn != GL_ALREADY_SIGNALED && waitReturn != GL_CONDITION_SATISFIED)
	    waitReturn = glClientWaitSync(gpuFence, GL_SYNC_FLUSH_COMMANDS_BIT, 1);

	texture.GetTexImage(pixels);

	flipImageVertically(width, height, pixels);

	if (!stbi_write_png("out.png", width, height, 4, pixels.data(), width * 4)) {
        std::cerr << "Failed to write image to file: " << "out.png" << std::endl;
    } else {
        std::cout << "Image successfully written to file: " << "out.png" << std::endl;
    }
}

void RayTraceManager::executeRenderOperation() {
	renderShader.Activate();

	texture.Bind();

	glUniform1i(glGetUniformLocation(renderShader.ID, "width"), width);
	glUniform1i(glGetUniformLocation(renderShader.ID, "height"), height);

	glDrawArrays(GL_TRIANGLES, 0, 6);
}
