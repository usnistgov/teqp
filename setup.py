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
            cmake_args += ['-T ClangCL']
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

init_template  = r'''import os, warnings, functools

# Bring all entities from the extension module into this namespace
from .teqp import *
from .teqp import _make_model

def get_datapath():
    """Get the absolute path to the folder containing the root of multi-fluid data"""
    return os.path.abspath(os.path.dirname(__file__)+"/fluiddata")

from .teqp import __version__

deprecated_top_level_functions = ["get_splus","get_pr","get_B2vir""get_B12vir","pure_VLE_T","extrapolate_from_critical","build_Psir_Hessian_autodiff","build_Psi_Hessian_autodiff","build_Psir_gradient_autodiff","build_d2PsirdTdrhoi_autodiff","get_chempotVLE_autodiff","get_dchempotdT_autodiff","get_fugacity_coefficients","get_partial_molar_volumes","trace_critical_arclength_binary","get_criticality_conditions","eigen_problem","get_minimum_eigenvalue_Psi_Hessian","get_drhovec_dT_crit","get_pure_critical_conditions_Jacobian","solve_pure_critical","mix_VLE_Tx","mixture_VLE_px","get_drhovecdp_Tsat","trace_VLE_isotherm_binary","get_drhovecdT_psat","trace_VLE_isobar_binary","get_dpsat_dTsat_isopleth","mix_VLLE_T","find_VLLE_T_binary"]

def deprecated_caller(model, *args, **kwargs):
    name = kwargs.pop('name123456')
    warnings.warn("Calling the top-level function " + name + " is deprecated and much slower than calling the same-named method of the model instance", FutureWarning)
    return getattr(model, name)(*args, **kwargs)
    
for f in deprecated_top_level_functions:
    globals()[f] = functools.partial(deprecated_caller, name123456=f)
    
# for factory_function in ['_make_vdW1']:

def tolist(a):
    if isinstance(a, list):
        return a
    else:
        try:
            return a.tolist()
        except:
            return a
        
def make_vdW1(a, b):
    AS = _make_model({"kind": "vdW1", "model": {"a": a, "b": b}})
    attach_model_specific_methods(AS)
    return AS
    
def vdWEOS1(*args):
    return make_vdW1(*args)
    
def make_model(*args):
    AS = _make_model(*args)
    attach_model_specific_methods(AS)
    return AS
    
def canonical_PR(Tc_K, pc_Pa, acentric, kmat=[]):
    j = {
        "kind": "PR",
        "model": {
            "Tcrit / K": tolist(Tc_K),
            "pcrit / Pa": tolist(pc_Pa),
            "acentric": tolist(acentric),
            "kmat": tolist(kmat)
        }
    }
    return make_model(j)
    
def canonical_SRK(Tc_K, pc_Pa, acentric, kmat=[]):
    j = {
        "kind": "SRK",
        "model": {
            "Tcrit / K": tolist(Tc_K),
            "pcrit / Pa": tolist(pc_Pa),
            "acentric": tolist(acentric),
            "kmat": tolist(kmat)
        }
    }
    return make_model(j)

def CPAfactory(spec):
    j = {
        "kind": "CPA",
        "model": spec
    }
    return make_model(j)
    
def PCSAFTEOS(names_or_coeffs, kmat = []):
    if isinstance(names_or_coeffs[0], SAFTCoeffs):
        coeffs = []
        for c in names_or_coeffs:
            coeffs.append({
                'name': c.name,
                'm': c.m,
                'sigma_Angstrom': c.sigma_Angstrom,
                'epsilon_over_k': c.epsilon_over_k,
                'BibTeXKey': c.BibTeXKey
            })
        spec = {'coeffs': coeffs, 'kmat': tolist(kmat)}
    else:
        spec = {'names': names_or_coeffs, 'kmat': tolist(kmat)}
    
    j = {
        "kind": "PCSAFT",
        "model": spec
    }
    return make_model(j)
    
def AmmoniaWaterTillnerRoth():
    return make_model({"kind": "AmmoniaWaterTillnerRoth", "model": {}})

def build_LJ126_TholJPCRD2016():
    return make_model({"kind": "LJ126_TholJPCRD2016", "model": {}})
    
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

toc = timeit.default_timer()

print('elapsed:', toc-tic, 'seconds')
