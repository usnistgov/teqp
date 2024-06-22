import sys, ctypes as ct, json, timeit

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
        uid = ct.c_longlong()
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
            return o.value
        else:
            raise ValueError(trim(errmsg))

if __name__ == '__main__':
    # Now load the library
    c = DLLCaller(full_path = '../../bld/Debug/libteqpc.dylib') # or .dll on windows
    model = {
      'kind': 'vdW1',
      'model': {'a': 1, 'b': 2}
    }
    uid = c.build_model(model)
    print(c.get_Arxy(uid=uid, NT=0,ND=1,T=300,rho=1,z=[1.0]))