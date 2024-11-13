import subprocess
from tqdm import tqdm
import os
import time
import numpy as np
import math


delta_g_to_kd = lambda x: math.exp(x / (0.00198720425864083 * 298.15))


def smiles_to_affinity(smiles, autodock='~/AutoDock-GPU/bin/autodock_gpu_128wi', protein_file='proteins/7cld/7cld.maps.fld', num_devices=4, starting_device=0):
    if not os.path.exists('ligands'):
        os.mkdir('ligands')
    if not os.path.exists('outs'):
        os.mkdir('outs')
    subprocess.run('rm core.*', shell=True, stderr=subprocess.DEVNULL)
    subprocess.run('rm outs/*.xml', shell=True, stderr=subprocess.DEVNULL)
    subprocess.run('rm outs/*.dlg', shell=True, stderr=subprocess.DEVNULL)
    subprocess.run('rm -rf ligands/*', shell=True)#, stderr=subprocess.DEVNULL)
    for device in range(starting_device, starting_device + num_devices):
        os.mkdir(f'ligands/{device}')
    device = starting_device
    for i, smile in enumerate(tqdm(smiles, desc='preparing ligands')):
        subprocess.Popen(f'obabel -:"{smile}" -O ligands/{device}/ligand{i}HASH{hash(smile)}.pdbqt -p 7.4 --partialcharge gasteiger --gen3d', shell=True, stderr=subprocess.DEVNULL)
        device += 1
        if device == starting_device + num_devices:
            device = starting_device
    while True:
        total = 0
        for device in range(starting_device, starting_device + num_devices):
            total += len(os.listdir(f'ligands/{device}'))
        if total == len(smiles):
            break
    time.sleep(1)
    print('running autodock..')
    subprocess.run('rm outs/*.xml', shell=True, stderr=subprocess.DEVNULL)
    subprocess.run('rm outs/*.dlg', shell=True, stderr=subprocess.DEVNULL)
    if len(smiles) == 1:
        subprocess.run(f'{autodock} -M {protein_file} -L ligands/0/ligand0.pdbqt -N outs/ligand0', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        ps = []
        for device in range(starting_device, starting_device + num_devices):
            ps.append(subprocess.Popen(f'{autodock} -M {protein_file} -B ligands/{device}/ligand*.pdbqt -N ../../outs/ -D {device + 1}', shell=True, stdout=subprocess.DEVNULL))
        stop = False
        while not stop: 
            stop = True
            for p in ps:
                if p.poll() is None:
                    time.sleep(1)
                    stop = False
    affins = [0 for _ in range(len(smiles))]
    for file in os.listdir('outs'):
        if file.endswith('.dlg'):
            content = open(f'outs/{file}').read()
            if '0.000   0.000   0.000  0.00  0.00' not in content:
                try:
                    affins[int(file.split('ligand')[1].split('HASH')[0])] = float([line for line in content.split('\n') if 'RANKING' in line][0].split()[3])
                except:
                    pass
    return [min(affin, 0) for affin in affins]


def run_autodock(smiles, multiply=5, protein_file=''):
    affins = np.array(smiles_to_affinity(smiles * multiply, protein_file=protein_file))
    affin_mins = affins.reshape((-1, len(smiles))).min(0)
    smile_to_affin = {smiles[i]: affin_mins[i] for i in range(len(smiles))}
    smile_to_pdb = {}
    for smile in smile_to_affin:
        for f_name in [f for f in os.listdir('outs') if ('.dlg' in f and str(hash(smile)) in f)]:
            f = open(f'outs/{f_name}', 'r').read()
            if f'Estimated Free Energy of Binding    =  {smile_to_affin[smile]}' in f:
                pdb = f.split(f'Estimated Free Energy of Binding    =  {smile_to_affin[smile]}')[1]
                pdb = pdb.split('DOCKED: REMARK                         _______ _______ _______ _____ _____    ______ ____')[1].split('DOCKED: ENDMDL')[0].strip()
                pdb = pdb.replace('DOCKED: ', '')
                pdb += '\nENDMDL'
                new_pdb = ''
                for line in pdb.split('\n'):
                    new_pdb += line[:66] + '\n'
                smile_to_pdb[smile] = new_pdb
            if f'Estimated Free Energy of Binding    = {smile_to_affin[smile]}' in f:
                pdb = f.split(f'Estimated Free Energy of Binding    = {smile_to_affin[smile]}')[1]
                pdb = pdb.split('DOCKED: REMARK                         _______ _______ _______ _____ _____    ______ ____')[1].split('DOCKED: ENDMDL')[0].strip()
                pdb = pdb.replace('DOCKED: ', '')
                pdb += '\nENDMDL'
                new_pdb = ''
                for line in pdb.split('\n'):
                    new_pdb += line[:66] + '\n'
                smile_to_pdb[smile] = new_pdb
    return {smile: delta_g_to_kd(smile_to_affin[smile]) for smile in smile_to_affin}, smile_to_pdb


def file_to_pdb_buffer(file):
    subprocess.run(f"obabel {file} -O mol.pdb", shell=True, stderr=subprocess.DEVNULL)
    f = open('mol.pdb', 'rb').read()
    os.remove('mol.pdb')
    return f


def canonicalize_pdb(pose):
    open('autodock_pose.pdb', 'wb').write(pose)
    subprocess.run('pymol -cq pymol_script.py', shell=True)
