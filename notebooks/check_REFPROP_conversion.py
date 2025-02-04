import teqp, os, numpy as np, glob, json

REFPROP = os.getenv('HOME') + '/REFPROP10sandboxFLDS'
assert(os.path.exists(REFPROP))
import ctREFPROP.ctREFPROP as ct 
RP = ct.REFPROPFunctionLibrary(REFPROP)
RP.SETPATHdll(REFPROP)

BIP, DEP = teqp.convert_HMXBNC(f'{REFPROP}/FLUIDS/HMX.BNC')
with open('BIPPP.json', 'w') as fp:
    fp.write(json.dumps(BIP))
with open('DEPP.json', 'w') as fp:
    fp.write(json.dumps(DEP))
    
FLDS = {}
hash2name = {}
for path in glob.glob(f'{REFPROP}/FLUIDS/*.FLD'):
    basename = os.path.basename(path)
    FLDS[basename] = teqp.convert_FLD(path, name=basename)
    # print(basename, FLDS[basename]["INFO"]["HASH"])
    hash2name[FLDS[basename]["INFO"]["HASH"]] = basename
    # if basename == 'D2O.FLD':
    #     with open(f'{basename}.json','w') as fp:
            # fp.write(json.dumps(FLDS[basename]))
    del basename

# Fluid files not in REFPROP
baddies = ['506609c0', 'fa4584a0', '22dd09c0', '7af7fd30']

for pair in BIP:
    hashes = pair['hash1'], pair['hash2']
    # Skip pures not in REFPROP
    acceptable = all([h not in baddies for h in hashes])
    if not acceptable: 
        continue
    names = [hash2name[h] for h in hashes]
    
    s = RP.SETUPdll(2, '*'.join(names), 'HMX.BNC', 'DEF')
    # print(RP.GETKTVdll(1,2))
    if RP.GETKTVdll(1,2).hmodij in ['BDW', 'BMW', 'BCC', 'BHA', 'BCH', 'BXH', 'BNH', 'BMH']:
        print('Skipping missing departure function: '+RP.GETKTVdll(1,2).hmodij)
        continue
    
    if s.ierr != 0:
        raise ValueError(s.herr)
    
    z0 = 0.5
    z = np.array([z0, 1-z0])
    TR, RHOR = RP.REDXdll(z)
    ALPHAR = RP.PHIXdll(0,0,1.0,1.0,z)
    
    model = teqp.make_model({
        "kind": "multifluid",
        "model": {
            "components": [FLDS[names[0]], FLDS[names[1]]],
            "BIP": BIP,
            "departure": DEP
        }
    })
    alphar = model.get_Ar00(TR, RHOR*1e3, z)
    
    Trerr = abs(TR/model.get_Tr(z)-1)
    rhorerr = abs(RHOR*1e3/model.get_rhor(z)-1)
    alpharerr = abs(ALPHAR/alphar-1)
    if Trerr > 1e-12:
        print(names, Trerr)
    if rhorerr > 1e-12:
        print(names, rhorerr)
    if alpharerr > 1e-12:
        print(RP.GETKTVdll(1, 2))
        print(names, alpharerr, ALPHAR, alphar)