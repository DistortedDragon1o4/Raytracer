#version 460 core

#define MAX_BOUNCES 8
#define MAX_SAMPLES 64

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
layout(binding = 0, rgba8) uniform image2D tex;

struct Vertex {
    vec3 coord;
    vec3 normal;
    int id;
};

struct Triangle {
    Vertex v1;
    Vertex v2;
    Vertex v3;
};

layout(binding = 0, std430) buffer TriangleArray {
    Triangle t[];
} triangles;

struct Light {
    vec3 coord;
    float radius;
    vec3 color;
    float intensity;
};

layout(binding = 1, std430) buffer LightArray {
    Light l[];
} lights;

struct Material {
    vec3 color;
    float subsurface;
    vec3 sheen_tint;
    float sheen;
    vec3 specular_tint;
    float specular_strength;
    float roughness;
    float metallic;
};

layout(binding = 2, std430) buffer MaterialArray {
    Material m[];
} materials;

uniform int width;
uniform int height;

uniform int meshSize;
uniform int lightSize;

uniform vec3 cameraPosition;
uniform mat4 matrix;

struct Ray {
    vec3 origin;
    vec3 direction;
};

uint murmurHash14(uvec4 src) {
    const uint M = 0x5bd1e995u;
    uint h = 1190494759u;
    src *= M;
    src ^= src >> 24u;
    src *= M;
    h *= M;
    h ^= src.x;
    h *= M;
    h ^= src.y;
    h *= M;
    h ^= src.z;
    h *= M;
    h ^= src.w;
    h ^= h >> 13u;
    h *= M;
    h ^= h >> 15u;
    return h;
}

// 1 output, 4 inputs
float hash14(vec4 src, inout float s) {
    uint h = murmurHash14(floatBitsToUint(src));
    s += uintBitsToFloat(h & 0x007fffffu | 0x3f800000u);
    return fract(uintBitsToFloat(h & 0x007fffffu | 0x3f800000u) - 1.0);
}

uvec2 murmurHash24(uvec4 src) {
    const uint M = 0x5bd1e995u;
    uvec2 h = uvec2(1190494759u, 2147483647u);
    src *= M;
    src ^= src >> 24u;
    src *= M;
    h *= M;
    h ^= src.x;
    h *= M;
    h ^= src.y;
    h *= M;
    h ^= src.z;
    h *= M;
    h ^= src.w;
    h ^= h >> 13u;
    h *= M;
    h ^= h >> 15u;
    return h;
}

// 2 outputs, 4 inputs
vec2 hash24(vec4 src, inout float s) {
    uvec2 h = murmurHash24(floatBitsToUint(src));
    s = (uintBitsToFloat(h & 0x007fffffu | 0x3f800000u) - 1.0).x;
    return uintBitsToFloat(h & 0x007fffffu | 0x3f800000u) - 1.0;
}

uvec3 murmurHash34(uvec4 src) {
    const uint M = 0x5bd1e995u;
    uvec3 h = uvec3(1190494759u, 2147483647u, 3559788179u);
    src *= M;
    src ^= src >> 24u;
    src *= M;
    h *= M;
    h ^= src.x;
    h *= M;
    h ^= src.y;
    h *= M;
    h ^= src.z;
    h *= M;
    h ^= src.w;
    h ^= h >> 13u;
    h *= M;
    h ^= h >> 15u;
    return h;
}

// 3 outputs, 4 inputs
vec3 hash34(vec4 src, inout float s) {
    uvec3 h = murmurHash34(floatBitsToUint(src));
    s = (uintBitsToFloat(h & 0x007fffffu | 0x3f800000u) - 1.0).x;
    return uintBitsToFloat(h & 0x007fffffu | 0x3f800000u) - 1.0;
}

bool raySphereIntersection(Ray ray, Light light, out float t) {
    // Vector from ray origin to sphere center
    vec3 oc = ray.origin - light.coord;

    // Coefficients for the quadratic equation: at^2 + bt + c = 0
    float a = dot(ray.direction, ray.direction); // a = |rayDirection|^2 (should be 1 if normalized)
    float b = 2.0 * dot(oc, ray.direction); // b = 2 * (oc · rayDirection)
    float c = dot(oc, oc) - light.radius * light.radius; // c = |oc|^2 - r^2

    // Discriminant of the quadratic equation
    float discriminant = b * b - 4.0 * a * c;

    // If discriminant is negative, no intersection
    if (discriminant < 0.0) {
        t = -1.0; // No intersection
        return false;
    }

    // Compute the two possible solutions
    float sqrtDiscriminant = sqrt(discriminant);
    float t1 = (-b - sqrtDiscriminant) / (2.0 * a);
    float t2 = (-b + sqrtDiscriminant) / (2.0 * a);

    // Find the closest positive intersection distance
    if (t1 > 0.0 && t2 > 0.0) {
        t = min(t1, t2); // Both intersections are in front of the ray
    } else if (t1 > 0.0) {
        t = t1; // Only t1 is in front of the ray
    } else if (t2 > 0.0) {
        t = t2; // Only t2 is in front of the ray
    } else {
        t = -1.0; // Both intersections are behind the ray
        return false;
    }

    return true; // Intersection found
}

bool rayTriangleIntersection(Ray ray, Triangle tri, out float t) {
    const float EPSILON = 1e-6;

    // Edge vectors
    vec3 edge1 = tri.v2.coord - tri.v1.coord;
    vec3 edge2 = tri.v3.coord - tri.v1.coord;

    // Calculate the determinant
    vec3 pvec = cross(ray.direction, edge2);
    float det = dot(edge1, pvec);

    // If the determinant is near zero, the ray is parallel to the triangle
    if (abs(det) < EPSILON) {
        return false;
    }

    float invDet = 1.0 / det;

    // Calculate the distance from v0 to the ray origin
    vec3 tvec = ray.origin - tri.v1.coord;

    // Calculate the U parameter and test bounds
    float u = dot(tvec, pvec) * invDet;
    if (u < 0.0 || u > 1.0) {
        return false;
    }

    // Calculate the V parameter and test bounds
    vec3 qvec = cross(tvec, edge1);
    float v = dot(ray.direction, qvec) * invDet;
    if (v < 0.0 || u + v > 1.0) {
        return false;
    }

    // Calculate the distance along the ray to the intersection point
    t = dot(edge2, qvec) * invDet;

    // If t is positive, the intersection point is in front of the ray origin
    return t > EPSILON;
}

bool getColorAndNormal(Ray ray, out vec3 pos, out int id, out vec3 light, out vec3 normal) {
    float k = 1024.0;
    for (int i = 0; i < lightSize; i++) {
        float t;
        if (raySphereIntersection(ray, lights.l[i], t)) {
            if (t <= k) {
                k = t;

                pos = (t * ray.direction) + ray.origin;
                id = 0;
                light = lights.l[i].color * lights.l[i].intensity;
                normal = vec3(0.0);
            }
        }
    }
    for (int i = 0; i < meshSize; i++) {
        float t;
        if (rayTriangleIntersection(ray, triangles.t[i], t)) {
            if (t <= k) {
                k = t;

                pos = (t * ray.direction) + ray.origin;

                mat3 interpolator = inverse(mat3(
                            triangles.t[i].v1.coord, triangles.t[i].v2.coord, triangles.t[i].v3.coord
                        ));

                vec3 multiplier = interpolator * pos;

                id = triangles.t[i].v1.id;
                light = vec3(0.0);
                normal = (multiplier.x * triangles.t[i].v1.normal) + (multiplier.y * triangles.t[i].v2.normal) + (multiplier.z * triangles.t[i].v3.normal);
            }
        }
    }
    return (k < 1024.0);
}

vec3 cosineWeightedRandomDirection(vec3 normal, vec4 seed, inout float s) {
    vec2 rand = hash24(seed, s);

    // Generate random point on a unit disk
    float theta = 2.0 * 3.141592653589793 * rand.x;
    float r = sqrt(rand.y);
    vec2 diskPoint = r * vec2(cos(theta), sin(theta));

    // Project point up to hemisphere to get cosine weighting
    float z = sqrt(1.0 - diskPoint.x * diskPoint.x - diskPoint.y * diskPoint.y);

    // Create orthonormal basis aligned with normal
    vec3 tangent, bitangent;

    // Method to create a basis from a normal - choose an arbitrary axis to align with
    if (abs(normal.x) > abs(normal.y)) {
        tangent = normalize(vec3(normal.z, 0.0, -normal.x));
    } else {
        tangent = normalize(vec3(0.0, -normal.z, normal.y));
    }
    bitangent = cross(normal, tangent);

    // Transform hemisphere sample to world space
    return normalize(tangent * diskPoint.x + bitangent * diskPoint.y + normal * z);
}

// Helper: Creates tangent/bitangent from normal
void orthonormalBasis(vec3 n, out vec3 t, out vec3 b) {
    // Pick arbitrary axis to align tangent (avoid singularity with up)
    vec3 up = abs(n.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    t = normalize(cross(up, n));
    b = cross(n, t);
}

vec3 GGXMicrofacetWeightedRandomDirection(vec3 normal, vec3 incoming, float roughness, vec4 seed, inout float s) {
    // Transform the incoming direction to local space (normal = 0,0,1)
    vec3 bitangent = normalize(cross(normal, abs(normal.x) > 0.1 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0)));
    vec3 tangent = cross(bitangent, normal);
    mat3 worldToLocal = mat3(tangent, bitangent, normal);

    vec3 localIncoming = worldToLocal * incoming;

    vec2 randomSample = hash24(seed, s);

    // GGX importance sampling
    float a = roughness * roughness;

    // Sample spherical coordinates (importance sampled)
    float phi = 2.0 * 3.141592653589793 * randomSample.x;
    float cosTheta = sqrt((1.0 - randomSample.y) / (1.0 + (a * a - 1.0) * randomSample.y));
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

    // Create half-vector in local space
    vec3 localHalfVector = vec3(
            sinTheta * cos(phi),
            sinTheta * sin(phi),
            cosTheta
        );

    // Transform back to world space
    mat3 localToWorld = transpose(worldToLocal);
    vec3 halfVector = localToWorld * localHalfVector;

    // Make sure the half vector is in the same hemisphere as the normal
    halfVector = normalize(halfVector * sign(dot(halfVector, normal)));

    // Reflect incoming direction about the microfacet normal
    vec3 outgoing = reflect(incoming, halfVector);

    // Make sure outgoing direction is in the same hemisphere as the normal
    if (dot(outgoing, normal) <= 0.0) {
        return cosineWeightedRandomDirection(normal, seed, s);
    }

    return outgoing;
}

float luminance(vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

float computeSpecularProbability(vec3 wi, vec3 normal, vec3 albedo, float metallic, float roughness) {
    // 1. Compute base reflectance (F0) - blend between dielectric and metal
    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    // 2. Calculate Fresnel at normal incidence (simplified for probability)
    float F0_luminance = dot(F0, vec3(0.2126, 0.7152, 0.0722)); // RGB to luminance

    // 3. Incorporate roughness into probability (rougher = less specular)
    float roughness_factor = 1.0 - (roughness * roughness); // Squared for perceptual linearity

    // 4. View-dependent Fresnel effect (Schlick approximation)
    float cosTheta = abs(dot(normal, wi));
    float fresnel = F0_luminance + (1.0 - F0_luminance) * pow(1.0 - cosTheta, 5);

    // 5. Final probability (clamped to avoid extremes)
    return clamp(fresnel * roughness_factor, 0.001, 0.999);
}

vec3 getBouncedDirection(vec3 normal, vec3 i, int id, out bool choice, out float weight, vec4 seed, inout float s) {
    float specularProbability = computeSpecularProbability(i, normal, materials.m[id].color, materials.m[id].metallic, materials.m[id].roughness);

    vec3 wo = vec3(0.0);

    if (hash14(seed, s) < specularProbability) {
        // Sample GGX specular
        wo = GGXMicrofacetWeightedRandomDirection(normal, i, materials.m[id].roughness, seed, s);
        choice = true;
        weight = (specularProbability > 0.0) ? 1.0 / specularProbability : 0.0;
    } else {
        // Sample Lambertian diffuse
        wo = cosineWeightedRandomDirection(normal, seed, s);
        choice = false;
        weight = (specularProbability < 1.0) ? 1.0 / (1.0 - specularProbability) : 0.0;
    }

    return wo;
}

bool traceRay(Ray ray, out int id, out vec3 light, out vec3 normal, out Ray newray, out bool choice, out float weight, inout float s) {
    vec3 pos = vec3(0.0);
    light = vec3(0.0);

    if (getColorAndNormal(ray, pos, id, light, normal)) {
        if (normal == vec3(0.0)) {
            return false;
        } else {
            Ray crntray = {
                    pos + (0.0000152587890625 * normal),
                    getBouncedDirection(normal, ray.direction, id, choice, weight, vec4(pos, s), s)
                };

            newray = crntray;

            return true;
        }
    } else {
        light = vec3(0.01);
        return false;
    }
}

float d_ggx(float alpha, float dot_nh) {
    highp float alpha2 = alpha * alpha;

    highp float sq_term = (dot_nh * dot_nh) * (alpha2 - 1.0) + 1.0;
    highp float denominator = max(3.141592653589793 * sq_term * sq_term, 0.0000152587890625);

    highp float ggx = alpha2 / denominator;

    return ggx;
}

float g_smith(float alpha, float dot_nx) {
    float k = alpha * 0.5;

    float denominator = max(((dot_nx * (1.0 - k)) + k), 0.0000152587890625);

    return 1.0 / denominator;
}

vec3 diffuseBRDF(vec3 incidence, vec3 reflected, vec3 normal, int material_id, vec3 color, float weight) {
    vec3 baseColor = materials.m[material_id].color;

    vec3 h = normalize(incidence + reflected);

    float f_schlick = materials.m[material_id].specular_strength + ((1.0 + materials.m[material_id].specular_strength) * pow(1 - dot(reflected, h), 5));

    float k_diffuse = (1.0 - materials.m[material_id].metallic);

    vec3 f_sheen = f_schlick * pow(materials.m[material_id].sheen, dot(reflected, h)) * materials.m[material_id].sheen_tint;

    vec3 f_diffuse = baseColor * k_diffuse * f_schlick;

    return f_diffuse * color;
}

vec3 specularBRDF(vec3 incidence, vec3 reflected, vec3 normal, int material_id, vec3 color, float weight) {
    vec3 baseColor = materials.m[material_id].color;

    vec3 h = normalize(incidence + reflected);

    float luminance = 0.3 * baseColor[0] + 0.6 * baseColor[1] + 0.1 * baseColor[2]; // luminance approx.

    // vec3 tint = luminance > 0.0 ? baseColor / luminance : vec3(1.0); // normalize lum. to isolate hue+sat
    // vec3 specular_color = mix(materials.m[material_id].specular_strength * 0.08 * mix(vec3(1.0), tint, materials.m[material_id].specular_tint), baseColor, materials.m[material_id].metallic);
    vec3 specular_color = mix(vec3(1.0), materials.m[material_id].specular_strength * materials.m[material_id].specular_tint, materials.m[material_id].metallic);
    // vec3 sheen_color = mix(vec3(1), tint, materials.m[material_id].sheen_tint);

    float f_schlick = materials.m[material_id].specular_strength + ((1.0 + materials.m[material_id].specular_strength) * pow(1 - dot(reflected, h), 5));

    float alpha = materials.m[material_id].roughness * materials.m[material_id].roughness;

    float g = g_smith(alpha, dot(normal, incidence)) * g_smith(alpha, dot(normal, reflected));

    vec3 f_specular = 0.25 * g * f_schlick * specular_color * max(dot(incidence, h), 0.0000152587890625) * max(dot(reflected, normal), 0.0000152587890625);

    return f_specular * color;
}

Ray getInitialRay(vec4 seed, inout float s) {
    // Get the pixel coordinates from the global invocation ID
    ivec2 pixelCoords = ivec2(gl_GlobalInvocationID.xy);
    float x = (float(pixelCoords.x) + hash14(seed, s)) / width; // Normalized x coordinate [0, 1]
    float y = (float(pixelCoords.y) + hash14(seed, s)) / height; // Normalized y coordinate [0, 1]

    // Create a ray direction in camera space
    vec3 rayDirectionCamera = (vec3((2.0 * x - 1.0), (2.0 * y - 1.0), -1.0));

    // Transform the ray direction to world space using the inverse of the camera matrix
    vec4 rayPointsTo4 = inverse(matrix) * vec4(rayDirectionCamera, 1.0);
    vec3 rayPointsTo = rayPointsTo4.xyz / rayPointsTo4.w;

    Ray initialray = {
            cameraPosition,
            normalize(rayPointsTo - cameraPosition)
        };

    return initialray;
}

vec3 getColor(inout float s) {
    vec3 finalColor = vec3(0.0);

    for (int j = 0; j < MAX_SAMPLES; j++) {
        vec3 color = vec3(1.0);
        vec3 light = vec3(0.0);

        Ray oldray = getInitialRay(vec4(cameraPosition + finalColor, s), s);

        for (int i = 0; i < MAX_BOUNCES; i++) {
            Ray newray;
            int material_id = 0;
            vec3 incomingLight = vec3(0.0);
            vec3 normal = vec3(0.0);

            bool choice = false;
            float weight = 0.0;

            bool r = traceRay(oldray, material_id, incomingLight, normal, newray, choice, weight, s);

            light += color * incomingLight;

            if (r) {
                if (choice) {
                    color = specularBRDF(newray.direction, -oldray.direction, normal, material_id, color, weight);
                } else {
                    color = diffuseBRDF(newray.direction, -oldray.direction, normal, material_id, color, weight);
                }
            } else {
                break;
            }

            oldray = newray;
        }

        finalColor += light;
    }

    return finalColor / vec3(MAX_SAMPLES);
}

vec3 filmicTonemap(vec3 color) {
    // Constants for the tonemapping curve
    float A = 0.15; // Shoulder strength
    float B = 0.50; // Linear strength
    float C = 0.10; // Linear angle
    float D = 0.20; // Toe strength
    float E = 0.02; // Toe numerator
    float F = 0.30; // Toe denominator
    float W = 11.2; // Linear white point

    // Apply the tonemapping curve
    vec3 x = color;
    vec3 result = (x * (A * x + vec3(C * B)) + vec3(D * E)) / (x * (A * x + vec3(B)) + vec3(D * F));
    result = result - E / F;

    // Normalize the result to the white point
    vec3 whiteScale = 1.0 / ((W * (A * W + vec3(C * B)) + vec3(D * E)) / (W * (A * W + vec3(B)) + vec3(D * F))) - vec3(E / F);
    return result * whiteScale;
}

void main() {
    float s = (gl_GlobalInvocationID.x * height) + gl_GlobalInvocationID.y;

    vec3 linearNonTonemappedColor = getColor(s);

    vec3 linearTonemappedColor = filmicTonemap(linearNonTonemappedColor);

    float gamma = 2.2;
    vec3 correctedColor = pow(linearTonemappedColor, vec3(1.0 / gamma));

    ivec2 pixelCoords = ivec2(gl_GlobalInvocationID.xy);
    imageStore(tex, pixelCoords, vec4(correctedColor, 1.0));
}
