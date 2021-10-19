"""
For twine auth, want a pypirc file with contents:

[pypi]
username = usernameusernameusername
password = XXXXXXXXXX
"""

import sys
import os
import subprocess
import glob

# Check presence of twine variables or config file
userc = False
if os.path.exists('pypirc'):
    userc = True
else:
    for k in ['TWINE_USERNAME','TWINE_PASSWORD']:
        if k not in os.environ:
            raise KeyError(f'You must set the twine environment variable {k}')

def build():
    for pyver in ['3.7','3.8','3.9']:
        # Build the wheel if it is not already built
        abbrv = pyver.replace('.', '')
        condaenv = f'conda-{pyver}'
        subprocess.check_call(f'conda create -y -n {condaenv} python={pyver}', shell=True)
        subprocess.check_call(f'conda activate {condaenv} && python -m pip install -U pip wheel', shell=True)
        try:
            subprocess.check_call(f'conda activate {condaenv} && python -m pip -vvv wheel .', shell=True)
        except BaseException as be:
            print(be)
        finally:
            subprocess.check_call(f'conda env remove -y -n {condaenv}',shell=True)

def upload():
    # Upload wheels
    if userc:
        twine_call = f'twine upload --config-file pypirc *.whl'
    else:
        twine_call = f'twine upload *.whl'
    subprocess.check_call(twine_call, shell=True)

steps = ['build','upload']
if not any([k in sys.argv for k in steps]):
    raise ValueError("need to pass at least one step to do: '+str(steps)")

if 'build' in sys.argv:
    build()
if 'upload' in sys.argv:
    upload()