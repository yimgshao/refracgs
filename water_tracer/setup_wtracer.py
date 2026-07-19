import os

from threedgrut.utils import jit


def setup_wtracer(conf):
    # find optix dev
    dependencies_dir = os.path.join(os.path.dirname(__file__), "dependencies")
    if os.path.exists(os.path.join(dependencies_dir, "optix-dev")):
        optix_path = os.path.join(dependencies_dir, "optix-dev")
    else:
        raise FileNotFoundError("Cound not find optix-dev.")

    include_paths = [
        os.path.join(os.path.dirname(__file__), "include"),
        os.path.join(optix_path, "include"),
    ]

    cflags = []
    cuda_flags = []

    source_files = [
        "src/WaterTracer.cpp",
        "bindings.cpp",
    ]

    source_paths = [os.path.join(os.path.dirname(__file__), fn) for fn in source_files]

    jit.load(
        name="libwtracer_cc",
        sources=source_paths,
        extra_cflags=cflags,
        extra_cuda_cflags=cuda_flags,
        extra_include_paths=include_paths,
    )