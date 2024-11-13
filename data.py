from utils import *
import selfies as sf
from sklearn.model_selection import train_test_split
import pickle
from secrets import token_hex
import shutil


cwd = os.getcwd()


class FasterDataLoader:
    def __init__(self, x, y, batch_size):
        self.x = x
        self.y = y
        self.idxs = np.arange(len(x))
        self.batch_size = batch_size

    def __iter__(self):
        np.random.shuffle(self.idxs)
        for i in range(0, len(self.x), self.batch_size):
            yield self.x[self.idxs[i:i+self.batch_size]], self.y[self.idxs[i:i+self.batch_size]]

    def __len__(self):
        return len(self.x)


class Data:
    def __init__(self, target):
        self.target = target
        self.l0_smiles, self.l0_y, self.l1_smiles, self.l1_y, self.l2_smiles, self.l2_y, self.l3_smiles, self.l3_y = self.get_raw_data()
        self.symbol_to_idx, self.idx_to_symbol = self.generate_vocab(self.l0_smiles + self.l1_smiles + self.l2_smiles + self.l3_smiles)
        self.encoding_cache = {}

    def load_ckpt(self, ckpt):
        self.l0_smiles, self.l0_y, self.l1_smiles, self.l1_y, self.l2_smiles, self.l2_y, self.l3_smiles, self.l3_y = pickle.load(open(ckpt, 'rb'))

    def get_raw_data(self):
        train_experimental_linear_reg(self.target)
        l0_smiles = [line.strip() for line in open('zinc250k.smi', 'r').readlines()][:200000]
        l0_y = predict_experimental_linear_reg(l0_smiles).tolist()
        l1_y = []
        l1_smiles = []
        l1_smiles = [line.strip() for line in open('zinc250k.smi', 'r').readlines()[:5]]
        l1_y = [min(smiles_to_affinity(smile * 30, protein_file=cwd + '/BAT.py/BAT-cmet-updated/docking_files/receptor.maps.fld' if self.target == 'cmet' else 'BAT.py/BAT-brd4/docking_files/LMCSS-5uf0_5uez_docked.maps.fld')) for smile in l1_smiles]
        l2_smiles = [line.strip() for line in open('zinc250k.smi', 'r').readlines()[5:10]]
        l2_y = []
        for id in (['2wd1', '4deg', '4dei', '4r1v', '5eob'] if self.target == 'cmet' else ['5ues', '5uet', '5uev', '5uez', '5uf0', '5uvs', '5uvy', '5uvz']):
            l2_y.append(smiles_to_affinity(l2_smiles, protein_file=f'{cwd}/BAT.py/BAT-{self.target}/docking_files/{id}.maps.fld'))
        l2_y = list(np.array(l2_y).mean(0))
        l3_smiles = [line.split()[2] for line in open('cmet_abfe_data.txt' if self.target == 'cmet' else 'brd42_abfe_data.txt', 'r') if line.startswith('result')]
        l3_y = [float(line.split()[1]) for line in open('cmet_abfe_data.txt' if self.target == 'cmet' else 'brd42_abfe_data.txt', 'r') if line.startswith('result')]
        return l0_smiles, l0_y, l1_smiles, l1_y, l2_smiles, l2_y, l3_smiles, l3_y

    def generate_vocab(self, smiles):
        selfies = [sf.encoder(smile) for smile in (smiles)]
        vocab = set()
        for s in selfies:
            vocab.update(sf.split_selfies(s))
        vocab = ['[nop]'] + list(sorted(vocab))
        symbol_to_idx = {s: i for i, s in enumerate(vocab)}
        idx_to_symbol = {i: s for i, s in enumerate(vocab)}
        return symbol_to_idx, idx_to_symbol

    def get_dataloaders(self, smiles, y, symbol_to_idx, max_len):
        x = [(self.encoding_cache[smile] if smile in self.encoding_cache else [symbol_to_idx[symbol] for symbol in sf.split_selfies(sf.encoder(smile)) if symbol in symbol_to_idx]) for smile in smiles]

        for i in range(len(x)):
            if smiles[i] not in self.encoding_cache:
                self.encoding_cache[smiles[i]] = x[i]

        y = [s for i, s in enumerate(y) if len(x[i]) < max_len]
        x = [s for s in x if len(s) < max_len]
        x = [(s + [0] * (max_len - len(s))) for s in x]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

        train_loader = FasterDataLoader(torch.tensor(x_train).cuda(), torch.tensor(y_train).view((-1, 1)).float().cuda(), 2**14)
        test_loader = FasterDataLoader(torch.tensor(x_test).cuda(), torch.tensor(y_test).view((-1, 1)).float().cuda(), 2**14)
        return train_loader, test_loader

    def get_data(self, max_len):
        l0_train, l0_test = self.get_dataloaders(self.l0_smiles, self.l0_y, self.symbol_to_idx, max_len)
        l1_train, l1_test = self.get_dataloaders(self.l1_smiles, self.l1_y, self.symbol_to_idx, max_len)
        l2_train, l2_test = self.get_dataloaders(self.l2_smiles, self.l2_y, self.symbol_to_idx, max_len)
        l3_train, l3_test = self.get_dataloaders(self.l3_smiles, self.l3_y, self.symbol_to_idx, max_len)
        return l0_train, l0_test, l1_train, l1_test, l2_train, l2_test, l3_train, l3_test, len(self.symbol_to_idx)
    
    def query_batch(self, smiles, l):
        if l == 0:
            self.l0_smiles.extend(smiles)
            self.l0_y.extend(predict_experimental_linear_reg(smiles))
            return self.l0_y[-len(smiles):]
        elif l == 1:
            self.l1_smiles.extend(smiles)
            dir = f'autodock/{token_hex(8)}'
            if not os.path.exists('autodock'):
                os.mkdir('autodock')
            os.mkdir(dir)
            os.chdir(dir)
            if len(smiles) == 1:
                self.l1_y.extend([min(smiles_to_affinity(smiles * 30, protein_file=cwd + '/BAT.py/BAT-cmet-updated/docking_files/receptor.maps.fld' if self.target == 'cmet' else 'BAT.py/BAT-brd4/docking_files/LMCSS-5uf0_5uez_docked.maps.fld'))])
            else:
                self.l1_y.extend(smiles_to_affinity(smiles, protein_file=cwd + '/BAT.py/BAT-cmet-updated/docking_files/receptor.maps.fld' if self.target == 'cmet' else 'BAT.py/BAT-brd4/docking_files/LMCSS-5uf0_5uez_docked.maps.fld'))
            os.chdir(cwd)
            shutil.rmtree(dir)
            return self.l1_y[-len(smiles):]
        elif l == 2:
            self.l2_smiles.extend(smiles)
            res = []
            dir = f'autodock/{token_hex(8)}'
            os.mkdir(dir)
            os.chdir(dir)
            for id in (['2wd1', '4deg', '4dei', '4r1v', '5eob'] if self.target == 'cmet' else ['5ues', '5uet', '5uev', '5uez', '5uf0', '5uvs', '5uvy', '5uvz']):
                if len(smiles) == 1:
                    res.append(smiles_to_affinity(smiles * 30, protein_file=f'{cwd}/docking_files/{id}.maps.fld')[:1])
                else:
                    res.append(smiles_to_affinity(smiles, protein_file=f'{cwd}/docking_files/{id}.maps.fld'))
            os.chdir(cwd)
            shutil.rmtree(dir)
            res = np.array(res).T
            self.l2_y.extend(res.mean(1))
            return self.l2_y[-len(smiles):]
        elif l == 3:
            self.l3_smiles.extend(smiles)
            for smile in smiles:
                try:
                    print('querying abfe', smile)
                    self.l3_y.append(cmet_abfe_explicit([smile], time_multiplier=0.3)[smile]['energy'] if self.target =='cmet' else abfe_explicit([smile], input_file='input-short-tev.in', steps={})[smile]['energy'])
                    print('abfe result', smile, self.l3_y[-1])
                except:
                    print('abfe failure')
                    os.chdir(cwd)
                    self.l3_y.append(0)
            return self.l3_y[-len(smiles):]
        else:
            raise Exception('invalid fidelity', l)
    
    def indices_to_smile(self, indices):
        return sf.decoder(''.join([self.idx_to_symbol[int(idx)] for idx in indices.cpu().flatten().tolist()]))
