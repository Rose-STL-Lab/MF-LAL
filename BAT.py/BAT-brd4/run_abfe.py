from utils import *
import subprocess
import os
import time
import sys
import codecs
from secrets import token_hex
import random


# def open_gpus(mem_cutoff=3000):
#     looking = False
#     out = subprocess.run('nvidia-smi', capture_output=True)
#     gpu_procs = []
#     for line in out.stdout.decode('utf-8').split('\n'):
#         if 'Processes:' in line:
#             looking = True
#             continue
#         if looking and 'MiB' in line:
#             gpu_procs.append((int(line.split()[1]), int(line.split()[7].split('M')[0])))
#     gpu_total = [0 for _ in range(8)]
#     for gpu_id, mem in gpu_procs:
#         gpu_total[gpu_id] += mem
#     open_gpus = []
#     for i in range(8):
#         if gpu_total[i] <= mem_cutoff:
#             open_gpus.append(i)
#     return open_gpus


def run_procs(to_run, devices):
    ds = [None for _ in devices]
    for path, cmd in to_run:
        try:
            os.chdir(path)
        except:
            continue
        i = ds.index(None)
        ds[i] = subprocess.Popen(f'CUDA_VISIBLE_DEVICES={devices[i]}, {cmd}', shell=True)
        while None not in ds:
            ds = [(d if d.poll() == None else None) for d in ds]
            time.sleep(10)
    while None in [d.poll() for d in ds if d]:
        time.sleep(10)
    os.chdir('/home/ubuntu/BAT.py/BAT-brd4')


hex_decoder = codecs.getdecoder('hex_codec')
smile = hex_decoder(sys.argv[1])[0].decode('utf-8').replace(' ', '+')

affin, pose = run_autodock([smile], multiply=100, protein_file='/home/ubuntu/BAT.py/BAT-brd4/docking_files/LMCSS-5uf0_5uez_docked.maps.fld')
print('docked affinity', affin[smile])
pose = pose[smile].encode('utf-8')
subprocess.call('obabel autodock_pose.pdb -O all-poses/pose0.pdb -d', shell=True)
if os.path.exists('all-poses/pose0.pdb'):
    pdb = open('all-poses/pose0.pdb', 'r').read()
    new = ''
    for line in pdb.split('\n'):
        if line.startswith('ATOM'):
            new += line.replace('UNL     1', 'LIG A  11').replace('           ', '      pose ') + '\n'
    open('all-poses/pose0.pdb', 'w').write(new)
else:
    print('No pose found')
    exit()

subprocess.call('rm -rf equil', shell=True)
subprocess.call('rm -rf fe', shell=True)
to_run = []
subprocess.call('python BAT.py -i input-sdr.in -s equil', shell=True)
to_run.append(('/home/ubuntu/BAT.py/BAT-brd4/equil/pose0', 'bash run-local.bash'))
run_procs(to_run, [2])

subprocess.call(f'python BAT.py -i input-sdr.in -s fe', shell=True)
to_run = []
for i in range(10):
    for letter in ['m', 'n']:
        to_run.append((f'/home/ubuntu/BAT.py/BAT-brd4/fe/pose0/rest/{letter}0{i}', 'bash run-local.bash'))
for i in range(12):
    for letter in ['e', 'v']:
        to_run.append((f'/home/ubuntu/BAT.py/BAT-brd4/fe/pose0/sdr/{letter}{i:02d}', 'bash run-local.bash'))
devices = [2, 3, 4, 5, 6, 7, 8]
run_procs(to_run, devices)

subprocess.call(f'python BAT.py -i input-sdr.in -s analysis', shell=True)