# Based on https://github.com/pybind/cmake_example

import os
import re
import sys
import platform
import subprocess
import shutil

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion

VERSION = '0.3.1.dev'
with open('interface/teqpversion.hpp','w') as fpver:
    fpver.write(f'#include <string>\nconst std::string TEQPVERSION = "{VERSION}";')

here = os.path.dirname(os.path.abspath(__file__))

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
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.', '--target', 'teqp'] + build_args, cwd=self.build_temp)

init_template  = r'''import os

# Bring all entities from the extension module into this namespace
from .teqp import * 

def get_datapath():
    """Get the absolute path to the folder containing the root of multi-fluid data"""
    return os.path.abspath(os.path.dirname(__file__)+"/fluiddata")

from .teqp import __version__
'''

def prepare():
    # Package up the fluid data files for the multi-fluid
    # model
    if os.path.exists('teqp'):
        shutil.rmtree('teqp')
    os.makedirs('teqp')
    shutil.copytree('mycp','teqp/fluiddata')

    # Make a temporary MANIFEST.in to avoid polluting the repository
    # since it only contains one line
    with open('MANIFEST.in','w') as fp:
        fp.write('recursive-include teqp *.json')
    
    with open('teqp/__init__.py', 'w') as fp:
        fp.write(init_template)

def teardown():
    shutil.rmtree('teqp')
    os.remove('MANIFEST.in')

try:
    prepare()
    setup(
        name='teqp',
        version=VERSION,
        author='Ian Bell and friends',
        author_email='ian.bell@nist.gov',
        description='Templated EQuation of state Package',
        long_description='',
        ext_modules=[CMakeExtension('teqp.teqp')], # teqp.teqp is the extension module that lives inside the teqp package
        packages=['teqp'],
        include_package_data=True,
        cmdclass=dict(build_ext=CMakeBuild),
        zip_safe=False,
    )
finally:
    teardown()