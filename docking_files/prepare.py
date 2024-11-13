import subprocess
import os
import shutil


for file in os.listdir('.'):
    if 'lig' not in file and file.endswith('.pdb'):
        receptor = file
        ligand = file.replace('.pdb', '_lig.pdb')
        subprocess.run(f'obabel {receptor} -O {receptor}', shell=True)
        subprocess.run(f'obabel {ligand} -O {ligand.replace(".pdb", ".mol2")}', shell=True)
        subprocess.run(f'obabel {ligand.replace(".pdb", ".mol2")} -O {ligand.replace(".pdb", ".pdbqt")}', shell=True)
        subprocess.run(f'PYTHONPATH=/home/ubuntu/limo/mgltools_x86_64Linux2_1.5.7/MGLToolsPckgs/:$PYTHONPATH python2 ~/limo/mgltools_x86_64Linux2_1.5.7/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py -r {receptor} -e True', shell=True)
        subprocess.run(f'PYTHONPATH=/home/ubuntu/limo/mgltools_x86_64Linux2_1.5.7/MGLToolsPckgs/:$PYTHONPATH python2 ~/limo/mgltools_x86_64Linux2_1.5.7/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_gpf4.py -p ligand_types=\'C,HD,N,A,NA,OA,F,S,Br,Cl\' -l {ligand.replace(".pdb", ".pdbqt")} -r {receptor.replace(".pdb", ".pdbqt")} -y', shell=True)
        subprocess.run(f'/home/ubuntu/limo/mgltools_x86_64Linux2_1.5.7/MGLToolsPckgs/AutoDockTools/Utilities24/autogrid4 -p {receptor.replace(".pdb", ".gpf")}', shell=True)