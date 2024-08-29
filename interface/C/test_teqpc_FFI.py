"""
A small script showing how to use the cffi library in Python to call the teqp
shared library
"""
from cffi import FFI
import json 

ffi = FFI()
ffi.cdef(open('teqpc.h').read().replace('EXPORT_CODE','').replace('CONVENTION',''))

# This next line will need to be changed to the absolute path of the library that was compiled
C = ffi.dlopen('bld/Debug/libteqpc.dylib')

handle = ffi.new("long long*")
buffer = ffi.new("char[]", ("?"*300).encode('ascii'))

spec = ffi.new("char[]", json.dumps({'kind':'vdW1', 'model': {'a': 1, 'b': 2}}).encode('ascii'))
C.build_model(spec, handle, buffer, 300)

print(handle[0])
print(ffi.string(buffer).decode('ascii'))