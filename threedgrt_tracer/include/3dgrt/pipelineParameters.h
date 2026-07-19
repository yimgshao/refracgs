// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// --- Modification Notice ---
// Modified by Yiming Shao
// Description: 
// Add gradient parameters for the origin and direction of rays.
// Modify rayWorldOrigin() and rayWorldDirection() for world space input.

#pragma once

#include <optix.h>

#include <3dgrt/particleDensity.h>
#include <3dgrt/pipelineDefinitions.h>
#include <3dgrt/tensorAccessor.h>

struct PipelineParameters {
    float4 rayToWorld[3];   ///< float3x4 ray to world transformation (row-major)
    PackedTensorAccessor32<float, 4> rayOrigin;    ///< ray origin
    PackedTensorAccessor32<float, 4> rayDirection; ///< ray direction

    const ParticleDensity* particleDensity; ///< position, scale, quaternions, density
    const float* particleRadiance;          ///< spherical harmonics coefficients
    const void* particleExtendedData;       ///< pipeline specific particle data
    int32_t* particleVisibility;       ///< pipeline specific particle data

    PackedTensorAccessor32<float, 4> rayRadiance;    ///< output integrated ray radiance
    PackedTensorAccessor32<float, 4> rayDensity;     ///< output integrated ray density
    PackedTensorAccessor32<float, 4> rayHitDistance; ///< output integrated ray hit distance
    PackedTensorAccessor32<float, 4> rayNormal;      ///< output integrated ray normal
    PackedTensorAccessor32<float, 4> rayHitsCount;   ///< output (only in AH pipeline) number of hits per ray

    OptixTraversableHandle handle;
    OptixAabb aabb;

    float minTransmittance;
    float hitMinGaussianResponse;
    static constexpr float hitMaxParticleSquaredDistance = 9.f; ///< by design
    float alphaMinThreshold;
    unsigned int sphDegree;

    uint2 frameBounds;
    unsigned int frameNumber;
    int gPrimNumTri;

    static constexpr unsigned int MaxNumHitPerTrace = 16;

#ifdef PARTICLE_PRIMITIVE_TYPE
    static constexpr bool CustomPrimitive = (PARTICLE_PRIMITIVE_TYPE == MOGPrimitiveTypes::MOGTracingCustom);
    static constexpr bool InstancePrimitive = (PARTICLE_PRIMITIVE_TYPE == MOGPrimitiveTypes::MOGTracingInstances);
    static constexpr bool SurfelPrimitive = (PARTICLE_PRIMITIVE_TYPE == MOGPrimitiveTypes::MOGTracingTriSurfel);
    static constexpr bool ClampedPrimitive = PARTICLE_PRIMITIVE_CLAMPED;
#endif

#ifdef PARTICLE_KERNEL_DEGREE
    static constexpr int ParticleKernelDegree = PARTICLE_KERNEL_DEGREE;
#endif

#ifdef __CUDACC__
    inline __device__ float3 rayWorldOrigin(const uint3& idx) const {
        return make_float3(rayOrigin[idx.z][idx.y][idx.x][0], 
                                    rayOrigin[idx.z][idx.y][idx.x][1], 
                                    rayOrigin[idx.z][idx.y][idx.x][2]);
    }

    inline __device__ float3 rayWorldDirection(const uint3& idx) const {
        return make_float3(rayDirection[idx.z][idx.y][idx.x][0], 
                                       rayDirection[idx.z][idx.y][idx.x][1], 
                                       rayDirection[idx.z][idx.y][idx.x][2]);
    } 

#endif
};

struct PipelineBackwardParameters : PipelineParameters {
    PackedTensorAccessor32<float, 4> rayRadianceGrad;    ///< integrated ray radiance gradient
    PackedTensorAccessor32<float, 4> rayDensityGrad;     ///< integrated ray density gradient
    PackedTensorAccessor32<float, 4> rayHitDistanceGrad; ///< integrated ray hit distance gradient
    PackedTensorAccessor32<float, 4> rayNormalGrad;      ///< integrated ray hit distance gradient

    ParticleDensity* particleDensityGrad; ///< output position, scale, quaternions, density gradient
    float* particleRadianceGrad;          ///< output spherical harmonics coefficients gradient
    PackedTensorAccessor32<float, 4> rayOriginGrad;      ///< output rays origin gradient
    PackedTensorAccessor32<float, 4> rayDirectionGrad;   ///< output rays direction gradient
};
