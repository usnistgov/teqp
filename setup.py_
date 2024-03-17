# Based on https://github.com/pybind/cmake_example

import os
import re
import sys
import platform
import subprocess
import shutil
import re
import timeit

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion

# VERSION is now read from teqpversion.hpp header file
match = re.search(r'TEQPVERSION = \"([0-9a-z.]+)\"\;', open('interface/teqpversion.hpp').read())
if match:
    VERSION = match.group(1)
else:
    raise ValueError("Unable to parse version string from interface/teqpversion.hpp")

here = os.path.dirname(os.path.abspath(__file__))

tic = timeit.default_timer()

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            # cmake_args += ['-T ClangCL']
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        
        # Config
        cmake_elements = ['cmake', ext.sourcedir] + cmake_args
        print('cmake config command:', ' '.join(cmake_elements))
        print('running from:', self.build_temp)
        subprocess.check_call(cmake_elements, cwd=self.build_temp, env=env)

        # Build
        build_elements = ['cmake', '--build', '.', '--target', 'teqp'] + build_args
        print('cmake build command:', ' '.join(build_elements))
        subprocess.check_call(build_elements, cwd=self.build_temp)

setup(
    name='teqp',
    version=VERSION,
    author='Ian Bell and friends',
    author_email='ian.bell@nist.gov',
    description='Templated EQuation of state Package',
    long_description='',
    ext_modules=[CMakeExtension('teqp.teqp')], # teqp.teqp is the extension module that lives inside the teqp package
    packages=['teqp','teqp.fluiddata.dev.fluids','teqp.fluiddata.dev.mixtures'],
    include_package_data=True,
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)
toc = timeit.default_timer()

print('elapsed:', toc-tic, 'seconds')
