#include "WaterTracer.h"

#include <fstream>
#include <iostream>
#include <vector> 
#include <cstring> 
#include <nvrtc.h>
#include <optix_stack_size.h>
#include <optix_function_table_definition.h>


// Code from 3dgrt tracer
namespace {

void contextLogCB(unsigned int level, const char* tag, const char* message, void* /*cbdata */) {
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
}

bool readSourceFile(std::string& str, const std::string& filename) {
    // Try to open file
    std::ifstream file(filename.c_str(), std::ios::binary);
    if (file.good()) {
        // Found usable source file
        std::vector<unsigned char> buffer = std::vector<unsigned char>(std::istreambuf_iterator<char>(file), {});
        str.assign(buffer.begin(), buffer.end());
        return true;
    }
    return false;
}

void getCuStringFromFile(std::string& cu, const char* filename) {
    // Try to get source code from file
    if (readSourceFile(cu, filename)) {
        return;
    }

    // Wasn't able to find or open the requested file
    throw std::runtime_error("Couldn't open source file " + std::string(filename));
}

void getPtxFromCuString(std::string& ptx,
                        const char* include_dir,
                        const char* optix_include_dir,
                        const char* cuda_include_dir,
                        const char* cu_source,
                        const char* name,
                        const char** log_string) {
    // Create program
    nvrtcProgram prog = 0;
    NVRTC_CHECK_ERROR(nvrtcCreateProgram(&prog, cu_source, name, 0, NULL, NULL));

    // Gather NVRTC options
    std::vector<const char*> options;

    std::string sample_dir;
    sample_dir = std::string("-I") + include_dir;
    options.push_back(sample_dir.c_str());

    // Collect include dirs
    std::vector<std::string> include_dirs;

    include_dirs.push_back(std::string("-I") + optix_include_dir);
    include_dirs.push_back(std::string("-I") + cuda_include_dir);

    for (const std::string& dir : include_dirs) {
        options.push_back(dir.c_str());
    }

    // Collect NVRTC options
    const char* compiler_options[] = {CUDA_NVRTC_OPTIONS};
    std::copy(std::begin(compiler_options), std::end(compiler_options), std::back_inserter(options));

    // JIT compile CU to PTX
    const nvrtcResult compileRes = nvrtcCompileProgram(prog, (int)options.size(), options.data());

    // Retrieve log output
    std::string g_nvrtcLog;
    size_t log_size = 0;
    NVRTC_CHECK_ERROR(nvrtcGetProgramLogSize(prog, &log_size));
    g_nvrtcLog.resize(log_size);
    if (log_size > 1) {
        NVRTC_CHECK_ERROR(nvrtcGetProgramLog(prog, &g_nvrtcLog[0]));
        if (log_string)
            *log_string = g_nvrtcLog.c_str();
    }
    if (compileRes != NVRTC_SUCCESS)
        throw std::runtime_error("NVRTC Compilation failed.\n" + g_nvrtcLog);

    // Retrieve PTX code
    size_t ptx_size = 0;
    NVRTC_CHECK_ERROR(nvrtcGetPTXSize(prog, &ptx_size));
    ptx.resize(ptx_size);
    NVRTC_CHECK_ERROR(nvrtcGetPTX(prog, &ptx[0]));

    // Cleanup
    NVRTC_CHECK_ERROR(nvrtcDestroyProgram(&prog));
}

const char* getInputData(const char* filename,
                         const char* include_dir,
                         const char* optix_include_dir,
                         const char* cuda_include_dir,
                         const char* name,
                         size_t& dataSize,
                         const char** log) {
    if (log)
        *log = NULL;

    std::string *ptx, cu;
    ptx = new std::string();

    getCuStringFromFile(cu, filename);
    getPtxFromCuString(*ptx, include_dir, optix_include_dir, cuda_include_dir, cu.c_str(), name, log);

    dataSize = ptx->size();
    return ptx->c_str();
}

} // namespace

// ----------------------------------------------------------------------------
//

WaterTracer::WaterTracer(std::string path, std::string cuda_path)
{
    // Initialize the OptiX API, loading all API entry points
    OPTIX_CHECK(optixInit());
    CUDA_CHECK(cudaFree(0));

    // Specify context options
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &contextLogCB;
    options.logCallbackLevel          = 3;

    // Associate a CUDA context (and therefore a specific GPU) with this
    // device context
    CUcontext cuCtx = 0; // zero means take the current context
    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));

    const int numPayloadValues = 4;
    createPipeline(context, path, cuda_path, &module, &pipeline, sbt, numPayloadValues);
}

WaterTracer::~WaterTracer() {
    // Release GPU memory (all buffers below are allocated by this class)
    if (d_vertices)      CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_vertices)));
    if (d_indices)       CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_indices)));
    if (d_output_buffer) CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_output_buffer)));
    if (d_temp_buffer)   CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer)));

    // Destroy shader binding table
    if (sbt.raygenRecord)        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.raygenRecord)));
    if (sbt.missRecordBase)      CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.missRecordBase)));
    if (sbt.hitgroupRecordBase)  CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.hitgroupRecordBase)));

    // Destroy Optix objects
    if (pipeline) OPTIX_CHECK(optixPipelineDestroy(pipeline));
    if (module)   OPTIX_CHECK(optixModuleDestroy(module));

    // Destroy context
    if (context)  OPTIX_CHECK(optixDeviceContextDestroy(context));
}


void WaterTracer::createPipeline(const OptixDeviceContext context,
                                 const std::string path,
                                 const std::string cuda_path,
                                 OptixModule* module,
                                 OptixPipeline* pipeline,
                                 OptixShaderBindingTable& sbt,
                                 int numPayloadValues) {
    //
    // Load programs
    //
    char log[2048];
    OptixPipelineCompileOptions pipeline_compile_options = {};
    {
        OptixModuleCompileOptions module_compile_options = {};
        module_compile_options.maxRegisterCount          = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        module_compile_options.optLevel                  = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
        module_compile_options.debugLevel                = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

        pipeline_compile_options.usesMotionBlur                   = false;
        pipeline_compile_options.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pipeline_compile_options.numPayloadValues                 = numPayloadValues;
        pipeline_compile_options.numAttributeValues               = 0;
        pipeline_compile_options.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
        pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
        pipeline_compile_options.usesPrimitiveTypeFlags           = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

        // include dir and shader path
        size_t inputSize              = 0;
        std::string kernel_name       = "shader";
        std::string shaderFile        = path + "/src/kernels/" + kernel_name + ".cu";
        std::string includeDir        = path + "/include";
        std::string optix_include_dir = path + "/dependencies/optix-dev/include";
        std::string cuda_include_dir  = cuda_path + "/include";

        const char* input = getInputData(shaderFile.c_str(), includeDir.c_str(), optix_include_dir.c_str(),
                                         cuda_include_dir.c_str(), kernel_name.c_str(), inputSize, (const char**)&log);
        size_t sizeof_log = sizeof(log);

        OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
            context, &module_compile_options, &pipeline_compile_options, input, inputSize, log, &sizeof_log, module));
    }

    //
    // Create program groups
    //
    OptixProgramGroup raygen_prog_group   = nullptr;
    OptixProgramGroup miss_prog_group     = nullptr;
    OptixProgramGroup hitgroup_prog_group = nullptr;

    {
        OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros

        OptixProgramGroupDesc raygen_prog_group_desc = {};
        raygen_prog_group_desc.kind                  = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module            = *module;
        raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
        size_t sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &raygen_prog_group_desc,
                                                1, // num program groups
                                                &program_group_options, log, &sizeof_log, &raygen_prog_group));

        OptixProgramGroupDesc miss_prog_group_desc = {};
        miss_prog_group_desc.kind                  = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module            = *module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
        sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &miss_prog_group_desc,
                                                1, // num program groups
                                                &program_group_options, log, &sizeof_log, &miss_prog_group));

        OptixProgramGroupDesc hitgroup_prog_group_desc = {};
        hitgroup_prog_group_desc.kind                  = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hitgroup_prog_group_desc.hitgroup.moduleCH            = *module;
        hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
        sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &hitgroup_prog_group_desc,
                                                1, // num program groups
                                                &program_group_options, log, &sizeof_log, &hitgroup_prog_group));
    }

    //
    // Link pipeline
    //
    {
        const uint32_t max_trace_depth = 1;

        std::vector<OptixProgramGroup> program_groups;
        program_groups.push_back(raygen_prog_group);
        program_groups.push_back(miss_prog_group);
        program_groups.push_back(hitgroup_prog_group);

        OptixPipelineLinkOptions pipeline_link_options = {};
        pipeline_link_options.maxTraceDepth            = max_trace_depth;
        pipeline_link_options.debugLevel               = OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT;

        size_t sizeof_log                              = sizeof(log);
        OPTIX_CHECK_LOG(optixPipelineCreate(context, &pipeline_compile_options, &pipeline_link_options,
                                            program_groups.data(), static_cast<unsigned int>(program_groups.size()),
                                            log, &sizeof_log, pipeline));

        OptixStackSizes stack_sizes = {};
        for (auto& prog_group : program_groups) {
            OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes));
        }

        uint32_t direct_callable_stack_size_from_traversal;
        uint32_t direct_callable_stack_size_from_state;
        uint32_t continuation_stack_size;
        OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, max_trace_depth,
                                               0, // maxCCDepth
                                               0, // maxDCDEpth
                                               &direct_callable_stack_size_from_traversal,
                                               &direct_callable_stack_size_from_state, &continuation_stack_size));
        OPTIX_CHECK(optixPipelineSetStackSize(*pipeline, direct_callable_stack_size_from_traversal,
                                              direct_callable_stack_size_from_state, continuation_stack_size,
                                              1 // maxTraversableDepth
                                              ));
    }

    //
    // Set up shader binding table
    //
    {
        CUdeviceptr raygen_record;
        const size_t raygen_record_size = sizeof(RayGenSbtRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raygen_record), raygen_record_size));
        RayGenSbtRecord rg_sbt;
        OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
        CUDA_CHECK(
            cudaMemcpy(reinterpret_cast<void*>(raygen_record), &rg_sbt, raygen_record_size, cudaMemcpyHostToDevice));

        CUdeviceptr miss_record;
        size_t miss_record_size = sizeof(MissSbtRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&miss_record), miss_record_size));
        MissSbtRecord ms_sbt;
        OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group, &ms_sbt));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(miss_record), &ms_sbt, miss_record_size, cudaMemcpyHostToDevice));

        CUdeviceptr hitgroup_record;
        size_t hitgroup_record_size = sizeof(HitGroupSbtRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hitgroup_record), hitgroup_record_size));
        HitGroupSbtRecord hg_sbt;
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group, &hg_sbt));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(hitgroup_record), &hg_sbt, hitgroup_record_size, cudaMemcpyHostToDevice));

        sbt.raygenRecord                = raygen_record;
        sbt.missRecordBase              = miss_record;
        sbt.missRecordStrideInBytes     = sizeof(MissSbtRecord);
        sbt.missRecordCount             = 1;
        sbt.hitgroupRecordBase          = hitgroup_record;
        sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
        sbt.hitgroupRecordCount         = 1;
    }
}



void WaterTracer::buildBVH(torch::Tensor vertices, torch::Tensor triangles)
{
    if (d_output_buffer) 
    {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_output_buffer)));
        d_output_buffer = 0;
    }
    gas_handle = 0;

    vertices = vertices.contiguous();
    triangles = triangles.contiguous();

    const int num_vertices = vertices.size(0);
    const int num_triangles = triangles.size(0);

    // Copy the geometry into class-owned buffers instead of borrowing the
    // torch tensor storage. The GAS references these buffers during traversal,
    // so they must stay valid independently of the caller-side tensors.
    const size_t vertices_bytes  = static_cast<size_t>(num_vertices) * 3 * sizeof(float);
    const size_t triangles_bytes = static_cast<size_t>(num_triangles) * 3 * sizeof(int);

    // Reallocate the owned buffers only when they need to grow.
    if (vertices_bytes > d_vertices_bytes) {
        if (d_vertices) CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_vertices)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vertices), vertices_bytes));
        d_vertices_bytes = vertices_bytes;
    }
    if (triangles_bytes > d_indices_bytes) {
        if (d_indices) CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_indices)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_indices), triangles_bytes));
        d_indices_bytes = triangles_bytes;
    }

    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d_vertices), vertices.data_ptr<float>(),
                               vertices_bytes, cudaMemcpyDeviceToDevice, 0));
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d_indices), triangles.data_ptr<int>(),
                               triangles_bytes, cudaMemcpyDeviceToDevice, 0));

    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    build_input.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
    build_input.triangleArray.vertexStrideInBytes = sizeof(float) * 3;
    build_input.triangleArray.numVertices         = num_vertices;
    build_input.triangleArray.vertexBuffers       = &d_vertices;

    build_input.triangleArray.indexFormat         = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    build_input.triangleArray.indexStrideInBytes  = sizeof(int) * 3;
    build_input.triangleArray.numIndexTriplets    = num_triangles;
    build_input.triangleArray.indexBuffer         = d_indices;

    static const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
    build_input.triangleArray.flags               = triangle_input_flags;
    build_input.triangleArray.numSbtRecords       = 1;

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
    accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        context,
        &accel_options,
        &build_input,
        1,
        &buffer_sizes));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer), buffer_sizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_output_buffer), buffer_sizes.outputSizeInBytes));

    OPTIX_CHECK(optixAccelBuild(
        context,
        0, // CUDA stream
        &accel_options,
        &build_input,
        1,
        d_temp_buffer,
        buffer_sizes.tempSizeInBytes,
        d_output_buffer,
        buffer_sizes.outputSizeInBytes,
        &gas_handle,
        nullptr,
        0));

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer)));
    d_temp_buffer = 0;
}

std::tuple<torch::Tensor, torch::Tensor> WaterTracer::trace(torch::Tensor rays_ori, torch::Tensor rays_dir) {
    rays_ori = rays_ori.contiguous();
    rays_dir = rays_dir.contiguous();
    const int num_rays = rays_ori.size(0);

    torch::Tensor hit_indices  = torch::full({num_rays}, -1, torch::dtype(torch::kInt32).device(torch::kCUDA));
    torch::Tensor hit_distances= torch::full({num_rays}, -1.0f, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    CUdeviceptr d_ray_ori     = reinterpret_cast<CUdeviceptr>(rays_ori.data_ptr<float>());
    CUdeviceptr d_ray_dir     = reinterpret_cast<CUdeviceptr>(rays_dir.data_ptr<float>());
    CUdeviceptr d_hit_indices = reinterpret_cast<CUdeviceptr>(hit_indices.data_ptr<int>());
    CUdeviceptr d_hit_dist    = reinterpret_cast<CUdeviceptr>(hit_distances.data_ptr<float>());

    LaunchParams params = {};
    params.ray_origins    = d_ray_ori;
    params.ray_directions = d_ray_dir;
    params.hit_indices    = d_hit_indices;
    params.hit_distances  = d_hit_dist;
    params.gas            = gas_handle;
    params.num_rays       = num_rays;

    CUdeviceptr d_params;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(LaunchParams)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_params), &params, sizeof(LaunchParams), cudaMemcpyHostToDevice));

    OPTIX_CHECK(optixLaunch(pipeline, 0, d_params, sizeof(LaunchParams), &sbt, num_rays, 1, 1));

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_params)));

    return std::tuple<torch::Tensor, torch::Tensor>(hit_indices, hit_distances);
}

