import subprocess, os, shutil

here = os.path.dirname(__file__)

def run():
    # Run doxygen (always)
    if os.path.exists(here+'/_static/'):
        shutil.rmtree(here+'/_static/')
    os.makedirs(here+'/_static')
    subprocess.check_call('doxygen Doxyfile', cwd=here+'/../..', shell=True)

    # Execute all the notebooks
    for path, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.ipynb') and '.ipynb_checkpoints' not in path:
                subprocess.check_output(f'jupyter nbconvert --allow-errors --to notebook --output {file} --execute {file}', shell=True, cwd=path)

if __name__ == '__main__':
    print('running sphinx_pre_run.py...')
    run()