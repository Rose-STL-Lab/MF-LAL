import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from torch import nn, optim
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from data import Data
import gpytorch
from botorch.models import ApproximateGPyTorchModel
from utils import *
import sys


class SurrogateGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, config):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(SurrogateGP, self).__init__(variational_strategy)
        self.deep_kernel = nn.Sequential(nn.Linear(config['latent_dim'], 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 32))
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel())

    def forward(self, x):
        x = self.deep_kernel(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        

class MFLAL(nn.Module):
    def __init__(self, vocab_len, config):
        super(MFLAL, self).__init__()
        self.config = config

        self.embedder = nn.Embedding(vocab_len, config['embedding_dim'], padding_idx=0)
        self.encoder = nn.Sequential(
            nn.Linear(config['embedding_dim'] * config['max_len'], 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2 * config['latent_dim'])
        )

        self.decoder1 = nn.Sequential(
            nn.Linear(config['latent_dim'], 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, vocab_len * config['max_len'])
        )

        self.decoder2 = nn.Sequential(
            nn.Linear(config['latent_dim'], 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, vocab_len * config['max_len'])
        )

        self.decoder3 = nn.Sequential(
            nn.Linear(config['latent_dim'], 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, vocab_len * config['max_len'])
        )

        self.decoder4 = nn.Sequential(
            nn.Linear(config['latent_dim'], 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, vocab_len * config['max_len'])
        )

        self.surrogate1 = ApproximateGPyTorchModel(SurrogateGP(torch.randn((1000, config['latent_dim'])).cuda(), config))
        self.surrogate2 = ApproximateGPyTorchModel(SurrogateGP(torch.randn((1000, config['latent_dim'])).cuda(), config))
        self.surrogate3 = ApproximateGPyTorchModel(SurrogateGP(torch.randn((1000, config['latent_dim'])).cuda(), config))
        self.surrogate4 = ApproximateGPyTorchModel(SurrogateGP(torch.randn((1000, config['latent_dim'])).cuda(), config))

        self.to_z2 = nn.Sequential(
            nn.Linear(config['latent_dim'] * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, config['latent_dim'] * 2)
        )

        self.to_z3 = nn.Sequential(
            nn.Linear(config['latent_dim'] * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, config['latent_dim'] * 2)
        )

        self.to_z4 = nn.Sequential(
            nn.Linear(config['latent_dim'] * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, config['latent_dim'] * 2)
        )

    def encode(self, x):
        x = self.encoder(self.embedder(x).view((len(x), -1)))
        mu = x[:, :self.config['latent_dim']]
        log_var = x[:, self.config['latent_dim']:]
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        if self.training:
            return mu + eps * std, mu, log_var
        else:
            return mu, mu, log_var
        
    def transform(self, mu, log_var, l):
        z = torch.cat([mu, log_var], 1)
        if l == 1:
            x = self.to_z2(z)
        elif l == 2:
            x = self.to_z3(z)
        elif l == 3:
            x = self.to_z4(z)
        mu = x[:, :self.config['latent_dim']]
        log_var = x[:, self.config['latent_dim']:]
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        if self.training:
            return mu + eps * std, mu, log_var
        else:
            return mu, mu, log_var
    
    def decode(self, z, l):
        if l == 0:
            return self.decoder1(z).view((len(z), self.config['max_len'], -1))
        elif l == 1:
            return self.decoder2(z).view((len(z), self.config['max_len'], -1))
        elif l == 2:
            return self.decoder3(z).view((len(z), self.config['max_len'], -1))
        elif l == 3:
            return self.decoder4(z).view((len(z), self.config['max_len'], -1))

    def forward(self, x, l, mask=None):
        if mask is None:
            mask = torch.ones((len(x)), dtype=bool).cuda()
        z, mu, log_var = self.encode(x)
        if l == 0:
            return self.decode(z, 0), self.surrogate1(z[mask]), z, mu, log_var
        elif l == 1:
            z, mu, log_var = self.transform(mu, log_var, 1)
            return self.decode(z, 1), self.surrogate2(z[mask]), z, mu, log_var
        elif l == 2:
            z, mu, log_var = self.transform(mu, log_var, 1)
            z, mu, log_var = self.transform(mu, log_var, 2)
            return self.decode(z, 2), self.surrogate3(z[mask]), z, mu, log_var
        elif l == 3:
            z, mu, log_var = self.transform(mu, log_var, 1)
            z, mu, log_var = self.transform(mu, log_var, 2)
            z, mu, log_var = self.transform(mu, log_var, 3)
            return self.decode(z, 3), self.surrogate4(z[mask]), z, mu, log_var
        

def loss_function(model, x, y, l):
    out, surrogate_out, z, mu, log_var = model(x, l, mask=(y.squeeze() != 0))
    reconstruction_loss = F.cross_entropy(out.flatten(end_dim=1), x.flatten())
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / (x.shape[0] * x.shape[1])
    if l == 0:
        mll_loss = -mll_l0(surrogate_out, y.squeeze()[y.squeeze() != 0])
    elif l == 1:
        mll_loss = -mll_l1(surrogate_out, y.squeeze()[y.squeeze() != 0])
    elif l == 2:
        mll_loss = -mll_l2(surrogate_out, y.squeeze()[y.squeeze() != 0])
    elif l == 3:
        mll_loss = -mll_l3(surrogate_out, y.squeeze()[y.squeeze() != 0])
    corr = np.corrcoef(surrogate_out.mean.detach().flatten().cpu().numpy(), y[y.squeeze() != 0].detach().flatten().cpu().numpy())[0][1]
    loss = reconstruction_loss + config['kld_weight'] * kld_loss + config['mll_weight'] * mll_loss
    return loss, reconstruction_loss, kld_loss, mll_loss, corr, out


def train_iter(dataloader, l):
    reconstructions = []
    reconstruction_percents = []
    klds = []
    mlls = []
    corrs = []
    for x, y in dataloader:
        optimizer.zero_grad()
        loss, reconstruction_loss, kld_loss, mll_loss, corr, out = loss_function(model, x, y, l)
        reconstructions.append(reconstruction_loss.item())
        reconstruction_percents.append(1 - (out.argmax(2) != x).any(1).float().mean().item())
        klds.append(kld_loss.item())
        mlls.append(mll_loss.item())
        corrs.append(corr)

        if torch.is_grad_enabled():
            loss.backward()
            optimizer.step()
    return np.mean(reconstructions), np.mean(reconstruction_percents), np.mean(klds), np.mean(mlls), np.mean(corrs)


def gaussian_likelihood(x, mu, log_var):
    std_dev = torch.exp(log_var).sqrt()
    return torch.exp(-0.5 * ((x - mu) / std_dev) ** 2) / (std_dev * np.sqrt(2 * np.pi))


def get_top_latent_points(model, l, target_distribution = None, M=30):
    z = torch.randn((M, 1, config['latent_dim'])).cuda().requires_grad_(True)
    optimizer = optim.Adam([z], lr=0.1)
    for i in range(100):
        optimizer.zero_grad()
        out = model(z)
        likelihood = 0
        if target_distribution is not None:
            target_mu, target_logvar = target_distribution
            likelihood = gaussian_likelihood(z, target_mu, target_logvar).sum(dim=2).mean(1).mean()

        sims = []
        for j in range(len(z)):
            sims.append(F.cosine_similarity(z[j].squeeze().unsqueeze(0).expand(M, -1), z.squeeze(), dim=1).mean())
        sims = torch.stack(sims)
        loss = (out.mean - config['ucb_beta'][l] * out.variance).mean() + config['l2_coef'][l] * (z ** 2).mean() - config['likelihood_coef'][l] * likelihood + config['sim_coef'][l] * sims.mean()

        loss.backward()
        optimizer.step()
    return z.detach(), out.mean.detach() - out.variance.detach(), out.variance.detach(), likelihood.item() if isinstance(likelihood, torch.Tensor) else likelihood
    

def is_valid(smile, l):
    valid = True
    mol = MolFromSmiles(smile)
    for ring in mol.GetRingInfo().AtomRings():
        if not (4 < len(ring) < 7):
            valid = False
            break
    return valid and smiles_to_qed([smile])[0] > config['qed_cutoff'] and smiles_to_sa([smile])[0] < config['sa_cutoff']


def acq_step(target_dstribution, l, M=30):
    if l == 0:
        surrogate = model.surrogate1
    elif l == 1:
        surrogate = model.surrogate2
    elif l == 2:
        surrogate = model.surrogate3
    elif l == 3:
        surrogate = model.surrogate4

    z, ucb, var, likelihood = get_top_latent_points(surrogate, l, target_distribution=target_dstribution, M=M)
    x = model.decode(z, l).argmax(2)
    out, surrogate_out, z, mu, log_var = model(x, l)
    smiles, dist = [data.indices_to_smile(s) for s in x], (mu.detach(), log_var.detach())
    all_valid = False
    posterior_vars = var.detach()
    while not all_valid:
        print(sum([(0 if is_valid(smile, l) else 1) for smile in smiles]), 'not valid, redoing')

        sz, sucb, svar, likelihood = get_top_latent_points(surrogate, l, target_distribution=target_dstribution, M=M)
        sx = model.decode(sz, l).argmax(2)
        _, _, sz, smu, slog_var = model(sx, l)
        ssmiles = [data.indices_to_smile(s) for s in sx]
        smu, slog_var = smu.detach(), slog_var.detach()
        j = 0

        all_valid = True
        for i in range(len(smiles)):
            if not is_valid(smiles[i], l):
                all_valid = False
                if is_valid(ssmiles[j], l) and ssmiles[j] not in smiles:
                    smiles[i] = ssmiles[j]
                    dist[0][i] = smu[j]
                    dist[1][i] = slog_var[j]
                    posterior_vars[i] = svar[j]
                j += 1
    print(smiles)
    return smiles, dist, posterior_vars


if __name__ == '__main__':

    target = sys.argv[1]
    if target not in ['cmet', 'brd4-2']:
        print('invalid target')
        exit()

    data = Data(target)
    l0_train, l0_test, l1_train, l1_test, l2_train, l2_test, l3_train, l3_test, vocab_len = data.get_data(60)

    config = {
        'latent_dim': 64,
        'lr': 0.000114869430262957,
        'embedding_dim': 64,
        'max_len': 60,
        'kld_weight': 0.08232923258333508,
        'mll_weight': 1.3247001146843511,
        'ucb_beta': [1, 1, 1, 1], # 0.4 works better if faster improvement is desired
        'l2_coef': [0.7054918003005689, 0.7054918003005689, 0.7054918003005689, 0.7054918003005689],
        'likelihood_coef': [0.06407401672042821, 0.06407401672042821, 0.06407401672042821, 0.06407401672042821],
        'sim_coef': [0.033837829719624596, 0.5, 4.0, 4.0],
        'train_iters': 64,
        'gammas': [0.1, 0.1, 0.1],
        'max_iters': 100, # backup in case the variance never goes below gamma
        'qed_cutoff': 0.2, # paper uses 0.4 and 4, but changed it to these numbers for faster generation
        'sa_cutoff': 6
    }

    writer = SummaryWriter(f'logs/mf-lal')
    queried_means = []
    current_fid = 0

    model = MFLAL(vocab_len, config).cuda()
    optimizer = optim.Adam(model.parameters() , lr=config['lr'])

    for al_iter in range(100000):
        l0_train, l0_test, l1_train, l1_test, l2_train, l2_test, l3_train, l3_test, vocab_len = data.get_data(config['max_len'])

        mll_l0 = gpytorch.mlls.VariationalELBO(model.surrogate1.likelihood, model.surrogate1.model, num_data=len(l0_train))
        mll_l1 = gpytorch.mlls.VariationalELBO(model.surrogate2.likelihood, model.surrogate2.model, num_data=len(l1_train))
        mll_l2 = gpytorch.mlls.VariationalELBO(model.surrogate3.likelihood, model.surrogate3.model, num_data=len(l2_train))
        mll_l3 = gpytorch.mlls.VariationalELBO(model.surrogate4.likelihood, model.surrogate4.model, num_data=len(l3_train))

        for i in range(config['train_iters']):
            train_iter(l0_train, 0)
            train_iter(l1_train, 1)
            train_iter(l2_train, 2)
            train_iter(l3_train, 3)

        smiles, target_distribution, posterior_vars = acq_step(None, 0)
        if current_fid > 0:
            smiles, target_distribution, posterior_vars = acq_step(target_distribution, 1)
        if current_fid > 1:
            smiles, target_distribution, posterior_vars = acq_step(target_distribution, 2)
        if current_fid > 2:
            smiles, target_distribution, posterior_vars = acq_step(target_distribution, 3)

        queried_vals = np.array(data.query_batch([smiles[0]], current_fid))

        queried_means.append(np.mean(queried_vals[queried_vals != 0]))
        writer.add_scalar(f'al_l{current_fid}/molecular_sim', np.mean([[tanimoto_similarity(a, b) for b in smiles] for a in smiles]), al_iter)
        writer.add_scalar(f'al_l{current_fid}/proportion_unique', len(set(smiles)) / len(smiles), al_iter)
        writer.add_scalar(f'al_l{current_fid}/queried_val', queried_means[-1], al_iter)
        writer.flush()

        print('posterior var', posterior_vars[0][0].item())

        # the posterior variance can be finicky, so I also added max_iters to just increment
        # current_fid after a fixed number of iterations
        if current_fid < 3 and (posterior_vars[0][0].item() < config['gammas'][current_fid] or len(queried_means) > config['max_iters']):
            current_fid += 1
            print(f'increasing fidelity to {current_fid}')
            queried_means = []