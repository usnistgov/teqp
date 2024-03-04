import os, warnings, functools

# Bring all entities from the extension module into this namespace
from .teqp import *
from .teqp import _make_model, _build_multifluid_mutant, _build_multifluid_ecs_mutant

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
    
def make_model(*args, **kwargs):
    """
    This function is in two parts; first the make_model function (renamed to _make_model in the Python interface)
    is used to make the model and then the model-specific methods are attached to the instance
    """
    AS = _make_model(*args, **kwargs)
    attach_model_specific_methods(AS)
    return AS

def vdWEOS(Tc_K, pc_Pa):
    j = {
        "kind": "vdW",
        "model": {
            "Tcrit / K": tolist(Tc_K),
            "pcrit / Pa": tolist(pc_Pa)
        }
    }
    return make_model(j)
    
def canonical_PR(Tc_K, pc_Pa, acentric, kmat=None):
    j = {
        "kind": "PR",
        "model": {
            "Tcrit / K": tolist(Tc_K),
            "pcrit / Pa": tolist(pc_Pa),
            "acentric": tolist(acentric),
            "kmat": tolist(kmat) if kmat is not None else None
        }
    }
    return make_model(j)
    
def canonical_SRK(Tc_K, pc_Pa, acentric, kmat=None):
    j = {
        "kind": "SRK",
        "model": {
            "Tcrit / K": tolist(Tc_K),
            "pcrit / Pa": tolist(pc_Pa),
            "acentric": tolist(acentric),
            "kmat": tolist(kmat) if kmat is not None else None
        }
    }
    return make_model(j)

def CPAfactory(spec):
    j = {
        "kind": "CPA",
        "model": spec
    }
    return make_model(j)
    
def PCSAFTEOS(coeffs, kmat=None):
    if isinstance(coeffs[0], SAFTCoeffs):
        coeffs_ = []
        for c in coeffs:
            coeffs_.append({
                'name': c.name,
                'm': c.m,
                'sigma_Angstrom': c.sigma_Angstrom,
                'epsilon_over_k': c.epsilon_over_k,
                'BibTeXKey': c.BibTeXKey
            })
        spec = {'coeffs': coeffs_, 'kmat': tolist(kmat) if kmat is not None else None}
    else:
        spec = {'names': coeffs, 'kmat': tolist(kmat) if kmat is not None else None}
    
    j = {
        "kind": "PCSAFT",
        "model": spec
    }
    return make_model(j)
    
def AmmoniaWaterTillnerRoth():
    return make_model({"kind": "AmmoniaWaterTillnerRoth", "model": {}})

def build_LJ126_TholJPCRD2016():
    return make_model({"kind": "LJ126_TholJPCRD2016", "model": {}})

def IdealHelmholtz(model):
    return make_model({"kind": "IdealHelmholtz", "model": model})
    
def build_multifluid_model(components, coolprop_root, BIPcollectionpath = "", flags = {}, departurepath=""):
    j = {
        "kind": "multifluid",
        "model": {
            "components": components,
            "root": coolprop_root,
            "BIP": BIPcollectionpath,
            "flags": flags,
            "departure": departurepath
        }
    }
    return make_model(j)

def build_multifluid_mutant(*args, **kwargs):
    AS = _build_multifluid_mutant(*args, **kwargs)
    attach_model_specific_methods(AS)
    return AS

def build_multifluid_ecs_mutant(*args, **kwargs):
    AS = _build_multifluid_ecs_mutant(*args, **kwargs)
    attach_model_specific_methods(AS)
    return AS
