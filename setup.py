from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension

extra_objects = ['/usr/local/lib/libmicrondla.so']
extra_compile_args = {"cxx": ['-O3']}
setup(
    name='tmdla',
    packages=find_packages(),
    ext_modules=[
        CppExtension(
            name='tmdla._c',
            sources=['tmdla/tmdla.cpp', 'tmdla/tmdla_util.cpp'],
            extra_objects=extra_objects,
            extra_compile_args=extra_compile_args
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
