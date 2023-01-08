from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='faclin',
      ext_modules=[cpp_extension.CUDAExtension('faclin_cuda', ['faclin.cu'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
