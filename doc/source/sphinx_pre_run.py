import subprocess, os, shutil

here = os.path.dirname(__file__)

def run():
    # Run doxygen (always)
    if os.path.exists(here+'/_static/'):
        shutil.rmtree(here+'/_static/')
    os.makedirs(here+'/_static')
    subprocess.check_call('doxygen Doxyfile', cwd=here+'/../..', shell=True)

    # subprocess.check_output(f'jupyter nbconvert --version', shell=True)
    for path, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.ipynb') and '.ipynb_checkpoints' not in path:
                subprocess.check_output(f'jupyter nbconvert --allow-errors --to notebook --output {file} --execute {file}', shell=True, cwd=path)
                # --ExecutePreprocessor.allow_errors=True      (this allows you to allow errors globally, but a raises-exception cell tag is better)