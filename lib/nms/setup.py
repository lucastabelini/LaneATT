from setuptools import setup

from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(name='nms', packages=['nms'],
        package_dir={'':'src'},
        ext_modules=[CUDAExtension('nms.details', ['src/nms.cpp', 'src/nms_kernel.cu'])],
        cmdclass={'build_ext': BuildExtension})
