#include <optix.h>
#include <optix_device.h>

struct LaunchParams {
    float3* ray_origins;
    float3* ray_directions;
    int*    hit_indices;
    float*  hit_distances;
    OptixTraversableHandle gas;
    int num_rays;
};

extern "C" {
__constant__ LaunchParams params;
}

extern "C" __global__ void __raygen__rg() {
    const uint3 idx = optixGetLaunchIndex();
    const int index = idx.x;

    const float3 origin = params.ray_origins[index];
    const float3 direction = params.ray_directions[index];

    unsigned int prim_id = 0;
    unsigned int t_hit_u = __float_as_uint(-1.0f);

    optixTrace(
        params.gas,
        origin,
        direction,
        0.0f,                // tmin
        1e16f,               // tmax
        0.0f,                // rayTime
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        0,                   // SBT offset
        1,                   // SBT stride
        0,                   // miss index
        prim_id,
        t_hit_u
    );

    params.hit_indices[index]  = static_cast<int>(prim_id);
    params.hit_distances[index]= __uint_as_float(t_hit_u);
}

extern "C" __global__ void __closesthit__ch() {
    const unsigned int prim_id = optixGetPrimitiveIndex();
    const float t_hit = optixGetRayTmax();

    optixSetPayload_0(prim_id);
    optixSetPayload_1(__float_as_uint(t_hit));
}

extern "C" __global__ void __miss__ms() {
    optixSetPayload_0(0);
    optixSetPayload_1(__float_as_uint(-1.0f));
}
