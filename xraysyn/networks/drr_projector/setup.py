from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='drr_projector_function',
    ext_modules=[
        CUDAExtension('drr_projector_function', [
            'drr_projector.cpp',
            'dp_nearest_cuda.cu',
            'dp_trilinear_cuda.cu',
        ], extra_compile_args={
            'cxx': ['-g'],
            'nvcc': [
                "-gencode", "arch=compute_61,code=sm_61",
                "-gencode", "arch=compute_70,code=sm_70",
                "-gencode", "arch=compute_75,code=sm_75"]
        })
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
