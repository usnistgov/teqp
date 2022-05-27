""" This script runs *inside* the docker container """

import subprocess, os, sys, shutil, glob, timeit

tic = timeit.default_timer()

# Wipe contents of output folder 
# This script is running inside the docker container, so this should be allowed
for g in glob.glob('/output/*'):
    if os.path.isdir(g):
        shutil.rmtree(g)
    else:
        os.remove(g)

EXE = '/teqp/build/catch_tests'

# Collect the list of tags to be run
all_tags = []
output = subprocess.run(f'{EXE} --list-tags', shell = True, stdout = subprocess.PIPE).stdout.decode('utf-8')
for il, line in enumerate(output.split('\n')[1::]):
    if not line or '[' not in line: continue
    tag = '[' + line.split('[')[1]
    all_tags.append(tag)

tag_times = {}
for tag in all_tags:
    root =  tag.replace('[', '').replace(']','') + '.txt'
    print(tag, ' --> ', root)

    cmd = f'timeout 60m valgrind --tool=memcheck --error-limit=no --track-origins=yes {EXE} ' + tag
    tic1 = timeit.default_timer()
    with open('/output/log_'+root,'w') as fp_stderr:
        with open('/output/err_'+root,'w') as fp_stdout:
            subprocess.run(cmd, shell = True, stdout = fp_stdout, stderr = fp_stderr)    
    toc1 = timeit.default_timer()
    tag_times[tag] = toc1-tic1

    # Copy a debugging file to the output if present
    src_memorytest = 'memorytest.txt'
    dest_memorytest = '/output/memorytest_'+root
    if os.path.exists(src_memorytest):
        shutil.move(src_memorytest, dest_memorytest)

    print(open('/output/log_'+root).readlines()[-1])

# Print times taken for each tag
print('Time (in sec) taken for each tag')
for k, v in tag_times.items():
    print(k, v)

# Store all the outputs in zip archive
os.makedirs('/output/errors')
os.makedirs('/output/ok')
for g in glob.glob('/output/log_*.txt'):
    test_name = os.path.split(g)[1].replace('log_','').replace('.txt','')
    errname = f'/output/err_{test_name:s}.txt'
    is_ok = '0 errors from 0 contexts' in open(g).read()
    dest = '/output/ok' if is_ok else '/output/errors'
    for name in [g, errname]:
        shutil.move(name, dest)
shutil.make_archive('/output/errors','zip','/output/errors')
shutil.make_archive('/output/ok','zip','/output/ok')

toc = timeit.default_timer()
print('elapsed time [s]:', toc-tic)