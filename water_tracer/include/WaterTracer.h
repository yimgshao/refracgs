#pragma once

#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <tuple>
#include <string>

#include "cuoptixMacros.h"

template <typename T>
struct SbtRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

struct RayGenData {
    // No data needed
};
typedef SbtRecord<RayGenData> RayGenSbtRecord;

struct MissData {
    // No data needed
};
typedef SbtRecord<MissData> MissSbtRecord;

struct HitGroupData {
    // No data needed
};

typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

struct LaunchParams {
    CUdeviceptr ray_origins;
    CUdeviceptr ray_directions;
    CUdeviceptr hit_indices;
    CUdeviceptr hit_distances;
    OptixTraversableHandle gas;
    int num_rays;
};


class WaterTracer {
public:
    WaterTracer(std::string path, std::string cuda_path);
    ~WaterTracer();
    std::tuple<torch::Tensor, torch::Tensor> trace(torch::Tensor rays_ori, torch::Tensor rays_dir);
    void buildBVH(torch::Tensor vertices, torch::Tensor triangles);

private:
    OptixDeviceContext context = nullptr;
    OptixTraversableHandle gas_handle = 0;

    OptixPipeline pipeline = nullptr;
    OptixShaderBindingTable sbt = {};
    OptixModule module = nullptr;

    // Class-owned copies of the BVH source geometry. The OptiX GAS references
    // the vertex/index buffers during traversal instead of copying them, so
    // these buffers must stay valid for the whole lifetime of the GAS (i.e.
    // they cannot borrow the storage of caller-side torch tensors).
    CUdeviceptr d_vertices = 0;
    CUdeviceptr d_indices  = 0;
    // Allocated sizes (in bytes) of the buffers above; reallocate only on growth.
    size_t d_vertices_bytes = 0;
    size_t d_indices_bytes  = 0;

    CUdeviceptr d_output_buffer = 0;
    CUdeviceptr d_temp_buffer = 0;

    void createPipeline(const OptixDeviceContext context,
                        const std::string path,
                        const std::string cuda_path,
                        OptixModule* module,
                        OptixPipeline* pipeline,
                        OptixShaderBindingTable& sbt,
                        int numPayloadValues);
};
