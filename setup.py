from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

extra_objects = ['/usr/local/lib/libmicrondla.so']
extra_compile_args = {"cxx": ['-O3']}
setup(
    name='tmdla',
    ext_modules=[
        CppExtension(
            name='tmdla',
            sources=['tmdla.cpp', 'tmdla_util.cpp'],
            extra_objects=extra_objects,
            extra_compile_args=extra_compile_args
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
