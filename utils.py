import subprocess
import os
os.environ['NUMEXPR_MAX_THREADS'] = '8'
from rdkit.Chem import MolFromSmiles, QED
from rdkit.Chem import AllChem
from sascorer import calculateScore
import numpy as np
import time
import math
import torch
import csv
from secrets import token_hex
import random
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.DataStructs import FingerprintSimilarity, TanimotoSimilarity
from sklearn.linear_model import Lasso


cwd = os.getcwd()

delta_g_to_kd = lambda x: math.exp(x / (0.00198720425864083 * 298.15))
kd_to_delta_g = lambda x: 0.00198720425864083 * 298.15 * math.log(x)

abfe_devices = [0,1,2,3,4,5,6,7]
experimental_linear_reg = None

cmet_steps = {'eq_steps1': 500000, 
         'eq_steps2': 15000000, 
         't_steps1': 25000, 
         't_steps2': 50000, 
         'e_steps1': 50000, 
         'e_steps2': 50000, 
         'v_steps1': 100000, 
         'v_steps2': 100000}

brd4_steps = {'eq_steps1': 500000,
         'eq_steps2': 15000000,
         'm_steps1': 500000,
         'm_steps2': 1000000,
         'n_steps1': 500000,
         'n_steps2': 1000000,
         'e_steps1': 250000,
         'e_steps2': 500000,
         'v_steps1': 500000,
         'v_steps2': 1000000}

    
def smiles_to_sa(smiles):
    vals = []
    for smile in smiles:
        vals.append(calculateScore(MolFromSmiles(smile)))
    return vals


def smiles_to_qed(smiles):
    vals = []
    for smile in smiles:
        vals.append(QED.qed(MolFromSmiles(smile)))
    return vals


def smiles_to_morgan(smiles):
    out = []
    for smile in smiles:
        out.append(AllChem.GetMorganFingerprintAsBitVect(MolFromSmiles(smile), 3, nBits=2048))
    return np.array(out)


def smiles_to_affinity(smiles, autodock='~/AutoDock-GPU/bin/autodock_gpu_128wi', protein_file=cwd + '/BAT.py/BAT-brd4/docking_files/LMCSS-5uf0_5uez_docked.maps.fld', num_devices=torch.cuda.device_count(), starting_device=0):
    time.sleep(random.random())
    if not os.path.exists('ligands'):
        os.mkdir('ligands')
    if not os.path.exists('outs'):
        os.mkdir('outs')
    subprocess.run('rm core.*', shell=True, stderr=subprocess.DEVNULL)
    subprocess.run('rm outs/*.xml', shell=True, stderr=subprocess.DEVNULL)
    subprocess.run('rm outs/*.dlg', shell=True, stderr=subprocess.DEVNULL)
    subprocess.run('rm -rf ligands/*', shell=True, stderr=subprocess.DEVNULL)
    for device in range(starting_device, starting_device + num_devices):
        os.mkdir(f'ligands/{device}')
    device = starting_device
    for i, smile in enumerate(smiles):
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
    time.sleep(0.1)
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


def autodock(smiles, multiply=256, protein_file=None):
    if protein_file:
        affins = np.array(smiles_to_affinity(smiles * multiply, protein_file=protein_file))
    else:
        affins = np.array(smiles_to_affinity(smiles * multiply))
    affin_mins = affins.reshape((-1, len(smiles))).min(0)
    affin_means = affins.reshape((-1, len(smiles))).mean(0)
    affin_stds = affins.reshape((-1, len(smiles))).std(0)
    smile_to_data = {smiles[i]: {'total_energy': affin_mins[i], 'mean_total_energy': affin_means[i], 'std_total_energy': affin_stds[i]} for i in range(len(smiles))}
    for smile in smile_to_data:
        for f_name in [f for f in os.listdir('outs') if ('.dlg' in f and str(hash(smile)) in f)]:
            f = open(f'outs/{f_name}', 'r').read()
            if f"Estimated Free Energy of Binding    =  {smile_to_data[smile]['total_energy']}" in f or f"Estimated Free Energy of Binding    = {smile_to_data[smile]['total_energy']}" in f:
                print(f_name)
                if f"Estimated Free Energy of Binding    =  {smile_to_data[smile]['total_energy']}" in f:
                    pdb = f.split(f"Estimated Free Energy of Binding    =  {smile_to_data[smile]['total_energy']}")[1]
                else:
                    pdb = f.split(f"Estimated Free Energy of Binding    = {smile_to_data[smile]['total_energy']}")[1]
                try:
                    smile_to_data[smile]['intermolecular_energy'] = float(f.split('(1) Final Intermolecular Energy')[1].split()[1].strip())
                    smile_to_data[smile]['internal_energy'] = float(f.split('(2) Final Total Internal Energy')[1].split()[1].strip())
                    smile_to_data[smile]['torsional_energy'] = float(f.split('(3) Torsional Free Energy')[1].split()[1].strip())
                    smile_to_data[smile]['unbound_energy'] = float(f.split('(4) Unbound System\'s Energy')[1].split()[1].strip())
                    smile_to_data[smile]['num_evals'] = float(f.split('Number of energy evaluations:')[1].split()[0].strip())
                except:
                    if 'intermolecular_energy' not in smile_to_data[smile]:
                        smile_to_data[smile]['intermolecular_energy'] = 0
                    if 'internal_energy' not in smile_to_data[smile]:
                        smile_to_data[smile]['internal_energy'] = 0
                    if 'torsional_energy' not in smile_to_data[smile]:
                        smile_to_data[smile]['torsional_energy'] = 0
                    if 'unbound_energy' not in smile_to_data[smile]:
                        smile_to_data[smile]['unbound_energy'] = 0
                    if 'num_evals' not in smile_to_data[smile]:
                        smile_to_data[smile]['num_evals'] = 0
                pdb = pdb.split('DOCKED: REMARK                         _______ _______ _______ _____ _____    ______ ____')[1].split('DOCKED: ENDMDL')[0].strip()
                pdb = pdb.replace('DOCKED: ', '')
                atoms = []
                for line in pdb.split('\n'):
                    if line.startswith('ATOM'):
                        try:
                            _, _, type, _, _, x, y, z, vdw, _, _, _ = line.split()
                            atoms.append((type, float(vdw), float(x), float(y), float(z)))
                        except ValueError:
                            pass
                # smile_to_data[smile]['atom_coords'] = atoms
                smile_to_data[smile]['number_of_atoms'] = int(f.split('Number of atoms:')[1].split()[0].strip())
                smile_to_data[smile]['number_of_rotatable_bonds'] = int(f.split('Number of rotatable bonds:')[1].split()[0].strip())
                pdb += '\nENDMDL'
                new_pdb = ''
                for line in pdb.split('\n'):
                    new_line = line[:66] + '\n'
                    if new_line.startswith('ATOM'):
                        new_line = new_line.replace('ATOM  ', 'HETATM').replace('\n', '').replace('UNL  ', 'LIG A') + f'          {new_line[12:14]}  \n'
                        new_pdb += new_line
                pdb = new_pdb
                open('autodock_pose.pdb', 'w').write(pdb)
                # subprocess.run('pymol -cq pymol_script.py', shell=True)
                subprocess.call(f'obabel autodock_pose.pdb -O autodock_pose.pdbqt -p 7.4', shell=True, stdout=subprocess.DEVNULL)
                break
    return smile_to_data


def run_abfe_procs(to_run, devices):
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


def abfe_explicit(smiles, time_multiplier=1.0, gpus=abfe_devices, steps=brd4_steps, input_file='input-sdr.in'):
    start_dir = os.getcwd()
    os.environ['AMBERHOME'] = '/home/ubuntu/amber22'
    os.environ['PATH'] = '/home/ubuntu/amber22/bin:' + os.environ['PATH'].replace('/home/ubuntu/amber22/bin:', '')
    os.environ['PYTHONPATH'] = '/home/ubuntu/amber22/lib/python3.8/site-packages'
    out = {}
    for smile in smiles:
        if not os.path.exists('abfe_runs'):
            os.mkdir('abfe_runs')
        run_dir = cwd + '/abfe_runs/' + token_hex(16)
        print(run_dir)
        os.mkdir(run_dir)
        subprocess.run(f'cp -r BAT.py/BAT-brd4-updated/* {run_dir}', shell=True)
        os.chdir(run_dir)
        lines = []
        for line in open(input_file, 'r'):
            for step in steps:
                if line.startswith(step):
                    line = line.split()
                    line[2] = str(int(int(steps[step]) * time_multiplier))
                    line = ' '.join(line) + '\n'
            lines.append(line)
        open(input_file, 'w').writelines(lines)
        subprocess.call('rm -rf equil', shell=True)
        subprocess.call('rm -rf fe', shell=True)
        to_run = []
        autodock([smile])
        subprocess.run('obabel autodock_pose.pdb -O all-poses/pose0.pdb -p 7.4 -xu', shell=True)
        new_pdb = []
        i = 0
        for line in open('all-poses/pose0.pdb', 'r'):
            if line.startswith('COMPND') or line.startswith('AUTHOR'):
                continue
            if line.startswith('HETATM') or line.startswith('ATOM'):
                element = line.split()[2]
                i += 1
                if len(str(i)) == 1:
                    line = line[:14] + str(i) + line[15:]
                else:
                    line = line[:14] + str(i) + line[16:]
                line = line.replace('ATOM  ', 'HETATM')
                line = line.replace('UNL  ', 'LIG A')
                if line.strip().endswith('*'):
                    print('replacing * with', element)
                    line = line[:-(len(element) + 3)] + element + '\n'
            new_pdb.append(line)
        open('all-poses/pose0.pdb', 'w').writelines(new_pdb)
        subprocess.run('~/.conda/envs/paprika/bin/obabel all-poses/pose0.pdb -O all-poses/pose0.pdb -d', shell=True)
        subprocess.run('~/.conda/envs/paprika/bin/obabel all-poses/pose0.pdb -O all-poses/pose0.pdb -p 7.4', shell=True)
        i = 0
        new_pdb = []
        for line in open('all-poses/pose0.pdb', 'r'):
            if line.startswith('COMPND') or line.startswith('AUTHOR'):
                continue
            if (line.startswith('HETATM') or line.startswith('ATOM')) and line[13] == 'H':
                line = line.replace('ATOM  ', 'HETATM')
                i += 1
                if len(str(i)) == 1:
                    line = line[:14] + str(i) + line[15:]
                else:
                    line = line[:14] + str(i) + line[16:]
            line = line[:76] + line[76:].upper()
            new_pdb.append(line)
        open('all-poses/pose0.pdb', 'w').writelines(new_pdb)
        
        subprocess.call(f'python BAT.py -i {input_file} -s equil', shell=True)
        to_run.append((run_dir + '/equil/pose0', 'bash run-local.bash'))
        run_abfe_procs(to_run, [gpus[0]])

        os.chdir(run_dir)
        subprocess.call(f'python BAT.py -i {input_file} -s fe', shell=True)
        to_run = []
        for i in range(10):
            for letter in ['t']:#['m', 'n']:
                to_run.append((f'{run_dir}/fe/pose0/rest/{letter}0{i}', 'bash run-local.bash'))
        for i in range(12):
            for letter in ['e', 'v']:
                to_run.append((f'{run_dir}/fe/pose0/sdr/{letter}{i:02d}', 'bash run-local.bash'))
        run_abfe_procs(to_run, gpus)

        os.chdir(run_dir)
        subprocess.call(f'python BAT.py -i {input_file} -s analysis', shell=True)

        out[smile] = {'energy': float(open(run_dir + '/fe/pose0/Results/Results.dat', 'r').read().split('Binding free energy;')[1].split()[0].replace(';', '').strip())}
        subprocess.run(f'rm -rf {run_dir}', shell=True)
    os.chdir(start_dir)
    return out


def load_bindingdb_data(file, columns):
    outs = []
    targets = set()
    for row in csv.reader(open(file, 'r'), delimiter='	'):
        if row[0] == 'BindingDB Reactant_set_id' or False in [bool(row[i].strip()) for i in columns]:
            continue
        out = []
        if row[columns[0]] and '<' not in row[columns[0]] and '>' not in row[columns[0]]:
            out.append(row[1])
            out.append(kd_to_delta_g(float(row[columns[0]]) / 1e9))
            out.append(row[columns[1]])
        targets.add(row[columns[1]])
        if len(out) == len(columns) + 1:
            outs.append(out)
    return outs, list(targets)


def tanimoto_similarity(a, b):
    return FingerprintSimilarity(GetMorganFingerprintAsBitVect(MolFromSmiles(a), 2), 
                                 GetMorganFingerprintAsBitVect(MolFromSmiles(b), 2), 
                                 metric=TanimotoSimilarity)


def tanimoto_similarity_from_fps(a, b):
    return FingerprintSimilarity(a, 
                                 b, 
                                 metric=TanimotoSimilarity)


def smiles_to_fps(smiles):
    return [GetMorganFingerprintAsBitVect(MolFromSmiles(smile), 2) for smile in smiles]


def cmet_abfe_explicit(smiles, time_multiplier=1.0, gpus=abfe_devices, steps=cmet_steps, input_file='input-sdr.in'):
    start_dir = os.getcwd()
    os.environ['AMBERHOME'] = '/home/ubuntu/amber22'
    os.environ['PATH'] = '/home/ubuntu/amber22/bin:' + os.environ['PATH'].replace('/home/ubuntu/amber22/bin:', '')
    os.environ['PYTHONPATH'] = '/home/ubuntu/amber22/lib/python3.8/site-packages'
    out = {}
    for smile in smiles:
        if not os.path.exists('abfe_runs'):
            os.mkdir('abfe_runs')
        run_dir = cwd + '/abfe_runs/' + token_hex(16)
        print(run_dir)
        os.mkdir(run_dir)
        subprocess.run(f'cp -r BAT.py/BAT-cmet-updated/* {run_dir}', shell=True)
        os.chdir(run_dir)
        lines = []
        for line in open(input_file, 'r'):
            for step in steps:
                if line.startswith(step):
                    line = line.split()
                    line[2] = str(int(int(steps[step]) * time_multiplier))
                    line = ' '.join(line) + '\n'
            lines.append(line)
        open(input_file, 'w').writelines(lines)
        subprocess.call('rm -rf equil', shell=True)
        subprocess.call('rm -rf fe', shell=True)
        to_run = []
        autodock([smile], protein_file=cwd + '/BAT.py/BAT-cmet/docking_files/receptor.maps.fld')
        subprocess.run('obabel autodock_pose.pdb -O all-poses/pose0.pdb -p 7.4 -xu', shell=True)
        new_pdb = []
        i = 0
        for line in open('all-poses/pose0.pdb', 'r'):
            if line.startswith('COMPND') or line.startswith('AUTHOR'):
                continue
            if line.startswith('HETATM') or line.startswith('ATOM'):
                element = line.split()[2]
                i += 1
                if len(str(i)) == 1:
                    line = line[:14] + str(i) + line[15:]
                else:
                    line = line[:14] + str(i) + line[16:]
                line = line.replace('ATOM  ', 'HETATM')
                line = line.replace('UNL  ', 'LIG A')
                if line.strip().endswith('*'):
                    print('replacing * with', element)
                    line = line[:-(len(element) + 3)] + element + '\n'
            new_pdb.append(line)
        open('all-poses/pose0.pdb', 'w').writelines(new_pdb)
        subprocess.run('obabel all-poses/pose0.pdb -O all-poses/pose0.pdb -d', shell=True)
        subprocess.run('obabel all-poses/pose0.pdb -O all-poses/pose0.pdb -p 7.4', shell=True)
        i = 0
        new_pdb = []
        for line in open('all-poses/pose0.pdb', 'r'):
            if line.startswith('COMPND') or line.startswith('AUTHOR'):
                continue
            if (line.startswith('HETATM') or line.startswith('ATOM')) and line[13] == 'H':
                line = line.replace('ATOM  ', 'HETATM')
                i += 1
                if len(str(i)) == 1:
                    line = line[:14] + str(i) + line[15:]
                else:
                    line = line[:14] + str(i) + line[16:]
            line = line[:76] + line[76:].upper()
            new_pdb.append(line)
        open('all-poses/pose0.pdb', 'w').writelines(new_pdb)
        
        subprocess.call('python BAT.py -i input-sdr.in -s equil', shell=True)
        to_run.append((run_dir + '/equil/pose0', 'bash run-local.bash'))
        run_abfe_procs(to_run, [gpus[0]])

        os.chdir(run_dir)
        subprocess.call(f'python BAT.py -i input-sdr.in -s fe', shell=True)
        to_run = []
        for i in range(16):
            for letter in ['m', 'n']:
                to_run.append((f'{run_dir}/fe/pose0/rest/{letter}{i:02d}', 'bash run-local.bash'))
        for i in range(12):
            for letter in ['e', 'v']:
                to_run.append((f'{run_dir}/fe/pose0/sdr/{letter}{i:02d}', 'bash run-local.bash'))
        run_abfe_procs(to_run, gpus)

        os.chdir(run_dir)
        subprocess.call(f'python BAT.py -i input-sdr.in -s analysis', shell=True)

        out[smile] = {'energy': -float(open(run_dir + '/fe/pose0/Results/Results.dat', 'r').read().split('Binding free energy')[1].split()[0].strip())}
        subprocess.run(f'rm -rf {run_dir}', shell=True)
    os.chdir(start_dir)
    return out

def train_experimental_linear_reg(target):
    global experimental_linear_reg
    rows, targets = load_bindingdb_data('cmet.tsv' if target == 'cmet' else 'brd4-2.tsv', [9, 6])
    smiles = []
    y = []
    for smile, activity, target in rows:
        if MolFromSmiles(smile):
            smiles.append(smile)
            y.append(activity)
    x = smiles_to_morgan(smiles)
    experimental_linear_reg = Lasso().fit(x, y) # better generalization than normal linear regression

def predict_experimental_linear_reg(smiles):
    return experimental_linear_reg.predict(smiles_to_morgan(smiles))