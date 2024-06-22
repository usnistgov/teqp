import sys, ctypes as ct, json, timeit, os

contents = '''
{
  "EOS": [
    {
      "BibTeX_CP0": "",
      "BibTeX_EOS": "Thol-THESIS-2015",
      "STATES": {
        "reducing": {
          "T": 1.32,
          "T_units": "LJ units",
          "rhomolar": 0.31,
          "rhomolar_units": "LJ units"
        }
      },
      "T_max": 1200,
      "T_max_units": "LJ units",
      "Ttriple": 290.25,
      "Ttriple_units": "LJ units",
      "alphar": [
        {
          "d": [4, 1, 1, 2, 2, 3, 1, 1, 3, 2, 2, 5],
          "l": [0, 0, 0, 0, 0, 0, 1, 2, 2, 1, 2, 1],
          "n": [0.52080730e-2, 0.21862520e+1, -0.21610160e+1, 0.14527000e+1, -0.20417920e+1, 0.18695286e+0, -0.62086250e+0, -0.56883900e+0, -0.80055922e+0, 0.10901431e+0, -0.49745610e+0, -0.90988445e-1],
          "t": [1.000, 0.320, 0.505, 0.672, 0.843, 0.898, 1.205, 1.786, 2.770, 1.786, 2.590, 1.294],
          "type": "ResidualHelmholtzPower"
        },
        {
          "beta": [0.625, 0.638, 3.91, 0.156, 0.157, 0.153, 1.16, 1.73, 383, 0.112, 0.119],
          "d": [1, 1, 2, 3, 3, 2, 1, 2, 3, 1, 1],
          "epsilon": [ 0.2053, 0.409, 0.6, 1.203, 1.829, 1.397, 1.39, 0.539, 0.934, 2.369, 2.43],
          "eta": [2.067, 1.522, 8.82, 1.722, 0.679, 1.883, 3.925, 2.461, 28.2, 0.753, 0.82],
          "gamma": [0.71, 0.86, 1.94, 1.48, 1.49, 1.945, 3.02, 1.11, 1.17, 1.33, 0.24],
          "n": [-0.14667177e+1, 0.18914690e+1, -0.13837010e+0, -0.38696450e+0, 0.12657020e+0, 0.60578100e+0, 0.11791890e+1, -0.47732679e+0, -0.99218575e-1, -0.57479320e+0, 0.37729230e-2],
          "t": [2.830, 2.548, 4.650, 1.385, 1.460, 1.351, 0.660, 1.496, 1.830, 1.616, 4.970],
          "type": "ResidualHelmholtzGaussian"
        }
      ],
      "gas_constant": 1.0,
      "gas_constant_units": "LJ units",
      "molar_mass": 1.0,
      "molar_mass_units": "LJ units",
      "p_max": 100000,
      "p_max_units": "LJ units",
      "pseudo_pure": false
    }
  ],
  "INFO":{
    "NAME": "LennardJones",
    "REFPROP_NAME": "LJF",
    "CAS": "N/A"
    }
}
'''

def trim(s):
    return s.raw.replace(b'\x00',b'').strip().decode('utf-8')

class DLLCaller():
    def __init__(self, full_path):
        if sys.platform.startswith('win'):
            loader_fcn = ct.WinDLL
        else:
            loader_fcn = ct.CDLL

        self.dll = loader_fcn(full_path)

    def _getfcn(self, DLL, fname):
        try:
            return getattr(DLL, fname)
        except BaseException as BE:
            return None

    def build_model(self, model):
        f = self._getfcn(self.dll, 'build_model')
        hrf = ct.create_string_buffer(json.dumps(model).encode('utf-8'))
        uid = ct.c_longlong(0)
        errmsg = ct.create_string_buffer(1000)
        errcode = f(hrf, ct.byref(uid), errmsg, len(errmsg))
        if errcode == 0:
            return uid
        else:
            raise ValueError(trim(errmsg))

    def get_Arxy(self, *, uid, NT, ND, T, rho, z):
        f = self._getfcn(self.dll, 'get_Arxy')
        NT = ct.c_int(NT)
        ND = ct.c_int(ND)
        T = ct.c_double(T)
        rho = ct.c_double(rho)
        molefrac = (len(z)*ct.c_double)(*z)
        Ncomp = len(z)
        o = ct.c_double()
        errmsg = ct.create_string_buffer(1000)
        tic = timeit.default_timer()
        errcode = f(uid, NT, ND, T, rho, molefrac, Ncomp, ct.byref(o), errmsg, len(errmsg))
        toc = timeit.default_timer()
        if errcode == 0:
            return o
        else:
            raise ValueError(trim(errmsg))

if __name__ == '__main__':
    # Now load the library
    c = DLLCaller(full_path = '../../bld/Debug/libteqpc.dylib')
    model = {
      'kind': 'vdW1',
      'model': {'a': 1, 'b': 2}
    }
    uid = c.build_model(model)
    print(c.get_Arxy(uid=uid, NT=0,ND=1,T=300,rho=1,z=[1.0]))

    model = {
      'kind': 'multifluid',
      'model': {'components': [json.loads(contents)], 'departure': [], 'BIP': []}
    }
    uid = c.build_model(model)
    print(c.get_Arxy(uid=uid, NT=0, ND=0,T=1.5,rho=0.3,z=[1.0]))
    print(c.get_Arxy(uid=uid, NT=0, ND=1,T=1.5,rho=0.3,z=[1.0]))

    with open('ljf.json','w') as fp:
      fp.write(contents)

    model = {
      'kind': 'multifluid',
      'model': {'components': [json.load(open('ljf.json'))], 'departure': [], 'BIP': []}
    }
    uid = c.build_model(model)
    os.remove('ljf.json')
    print(c.get_Arxy(uid=uid, NT=0, ND=0,T=1.5,rho=0.3,z=[1.0]))
    print(c.get_Arxy(uid=uid, NT=0, ND=1,T=1.5,rho=0.3,z=[1.0]))