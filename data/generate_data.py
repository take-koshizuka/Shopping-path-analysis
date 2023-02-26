import h5py
import yaml
import csv
import json
import argparse
import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical
from torch.distributions.geometric import Geometric
from copy import deepcopy
from util import create_graph_and_product, to_tensor, save_h5py

class MaskedSoftmax:
    def __init__(self):
        self.softmax = torch.nn.Softmax(1)
    
    def forward(self, x, mask=None):
        """
        Performs masked softmax, as simply masking post-softmax can be
        inaccurate
        :param x: [batch_size, num_items]
        :param mask: [batch_size, num_items]
        :return:
        """
        if mask is not None:
            mask = mask.float()
        if mask is not None:
            x_masked = x * mask + (1 - 1 / mask)
        else:
            x_masked = x
        x_max = x_masked.max(1)[0]
        x_exp = (x - x_max.unsqueeze(-1)).exp()
        if mask is not None:
            x_exp = x_exp * mask.float()
        return x_exp / x_exp.sum(1).unsqueeze(-1)

class AttractionModel:
    def __init__(self, a_ikt, product, delta_is, delta_ib, w_v) -> None:
        """
            aikt (torch.Tensor): (N, K)
            product (): v -> (K,)
        """
        self.a_ikt = a_ikt #(N, K)
        self.N = a_ikt.size(0)
        self.K = a_ikt.size(1)
        self.product = product #(V, K)
        self.delta_is = delta_is.repeat(self.K, 1).T
        self.delta_ib = delta_ib.repeat(self.K, 1).T
        self.w_v = w_v
        self.V = self.product.size(0)
        assert self.product.size(1) == self.K
        self.A_ijt = self.cal_zone_attraction(self.a_ikt) # (N, V)
        # self.G_ijt = 
        self.log = dict(a_ikt=[deepcopy(self.a_ikt)])
    def I(self, x):
        return torch.stack([ self.product[v] for v in x ]) #(N,K)
    
    def cal_zone_attraction(self, a_ikt):
        return torch.log(torch.matmul(torch.exp(a_ikt), self.product.T)+1.0)

    def step(self, checkout, x_it, B_ikt, S_it, mask):
        # checkout: (N, K)
        non_checkout = torch.logical_not(checkout)
        mask_checkout = torch.logical_and(checkout, mask.repeat(self.K, 1).T)
        mask_non_checkout = torch.logical_and(non_checkout, mask.repeat(self.K, 1).T)
        self.a_ikt[mask_non_checkout] = self.a_ikt[mask_non_checkout] + self.delta_ib[mask_non_checkout] * B_ikt[mask_non_checkout] + \
                                                    self.delta_is[mask_non_checkout] * self.I(x_it)[mask_non_checkout]
        self.a_ikt[mask_checkout] = self.a_ikt[mask_checkout] + self.w_v * S_it.repeat(self.K, 1).T[mask_checkout]
        self.A_ijt = self.cal_zone_attraction(self.a_ikt) #(N, V)
        # self.G_ijt = 
        self.log['a_ikt'].append(deepcopy(self.a_ikt))
        return self.a_ikt

class VisitModel:
    def __init__(self, x_it, graph, rate, Z_j, kappa, gamma_v) -> None:
        self.x_it = x_it #(N, )
        self.N = self.x_it.size(0)
        self.graph = graph
        self.V = graph.V
        self.rho_jt = self.rho(self.x_it)
        self.acc_N = 0
        self.rate = rate
        self.Z_j = Z_j.repeat(self.N, 1)
        self.kappa = kappa.repeat(self.V, 1).T
        self.gamma_v = gamma_v
        self.T_0 = torch.zeros(self.N)
        self.log = dict(x_it=[], rho_jt=[])
        self.softmax = MaskedSoftmax()

    def visit(self, t):
        n = torch.poisson(torch.tensor(self.rate))
        n = min(int(n), self.N - self.acc_N)
        self.x_it[self.acc_N: self.acc_N + n] = self.graph.init
        self.T_0[self.acc_N: self.acc_N + n] = t
        self.acc_N += min(n, self.N - self.acc_N)
        self.rho_jt = self.rho(self.x_it)
        self.log['x_it'].append(deepcopy(self.x_it))
        self.log['rho_jt'].append(deepcopy(self.rho_jt))

    def rho(self, x):
        N_t = self.mask().sum()
        counter = torch.bincount(x, minlength=self.V) / N_t
        return counter

    def mask(self):
        return torch.logical_and(self.x_it != self.graph.outside, self.x_it != self.graph.checkout)
    
    def step(self, G_ijt, noise_v):
        u_v = self.Z_j + self.kappa * G_ijt + self.gamma_v * self.rho_jt.repeat(self.N, 1) + noise_v #(N, V)
        # idxs = self.graph.Adj[self.xit] #(N, V)
        
        self.x_it[self.x_it == self.graph.checkout] = self.graph.outside
        mask = self.graph.adj[self.x_it]
        p = self.softmax.forward(u_v, mask=mask)

        mask = self.mask()
        self.x_it[mask] = Categorical(p[mask]).sample()
        self.rho_jt = self.rho(self.x_it)

        return self.x_it

class ShopModel:
    def __init__(self, N, alpha_is, beta_is, w_s, gamma_s, tau_shop, tau_pass) -> None:
        self.N = N
        self.H_it = torch.zeros(self.N) #(N,)
        self.S_it = torch.zeros(self.N)
        self.log = dict(H_it=[torch.zeros(self.N)], S_it=[torch.zeros(self.N)])

        self.alpha_is = alpha_is
        self.beta_is = beta_is
        self.w_s = w_s
        self.gamma_s = gamma_s
        self.tau_shop = tau_shop
        self.tau_pass = tau_pass
    
    def step(self, A_ijt, x_it, T_it, rho_jt, noise_s, mask):
        self.N = T_it.size(0)
        u_s = self.alpha_is + self.beta_is * A_ijt[torch.arange(self.N), x_it] + \
                self.w_s * T_it + self.gamma_s * rho_jt[x_it] + noise_s # (N, )
        
        self.H_it = torch.bernoulli(torch.sigmoid(u_s))
        self.H_it[torch.logical_not(mask)] = 0.

        self.S_it = torch.zeros(self.N)
        self.S_it[self.H_it == 1] = Geometric(logits=self.tau_shop[x_it[self.H_it == 1]]).sample()
        self.S_it[self.H_it == 0] = Geometric(logits=self.tau_pass[x_it[self.H_it == 0]]).sample()
        self.S_it[torch.logical_not(mask)] = 0.

        self.log['H_it'].append(deepcopy(self.H_it))
        self.log['S_it'].append(deepcopy(self.S_it))
        return self.H_it, self.S_it

class BuyModel:
    def __init__(self, N, product, alpha_ib, beta_ib, gamma_b, w_b) -> None:
        self.N = N
        self.product = product #(V, K)
        self.K = product.size(1)
        self.B_ikt = torch.zeros(self.N, self.K) #(N, K)
        self.checkout = torch.zeros(self.N, self.K) #(N, K)
        self.log = dict(B_ikt=[deepcopy(self.B_ikt)])

        self.alpha_ib = alpha_ib.repeat(self.K, 1).T
        self.beta_ib = beta_ib.repeat(self.K, 1).T
        self.gamma_b = gamma_b
        self.w_b = w_b

    def I(self, x):
        return torch.stack([ self.product[v] for v in x ])

    def step(self, a_ikt, T_it, rho_jt, H_it, x_it, noise_b, mask):
        u_b = self.alpha_ib + self.beta_ib * a_ikt + self.w_b * T_it.repeat(self.K, 1).T + \
                    self.gamma_b * rho_jt[x_it].repeat(self.K, 1).T + noise_b #(N, K)

        self.B_ikt = torch.zeros(self.N, self.K)
        self.B_ikt[H_it == 1] = torch.bernoulli(torch.sigmoid(u_b[H_it == 1]))
        self.B_ikt[H_it == 0] = 0.
        self.B_ikt = self.B_ikt * self.I(x_it)
        self.B_ikt[torch.logical_not(mask)] = 0.
        self.checkout = torch.logical_or(self.checkout, self.B_ikt)
        self.log['B_ikt'].append(deepcopy(self.B_ikt))
        return self.B_ikt

class TimeModel:
    def __init__(self, T_it):
        self.T_it = T_it

    def step(self, mask):
        self.T_it[mask] = self.T_it[mask] + 1.

def generate_data(config):
    N = config['N']
    K = config['K']
    V = config['V']
    dist = config['dist']
    T_max = config['T_max']
    w_v = config['w_v']
    w_s = config['w_s']
    w_b = config['w_b']
    gamma_v = config['gamma_v']
    gamma_s = config['gamma_s']
    gamma_b = config['gamma_b']

    # sampling parameters
    kappa = torch.exp(torch.normal(**to_tensor(dist['kappa']),size=(N,)))
    alpha_is = torch.normal(**to_tensor(dist['alpha_is']),size=(N,))
    alpha_ib = torch.normal(**to_tensor(dist['alpha_ib']),size=(N,))
    beta_is = torch.normal(**to_tensor(dist['beta_is']),size=(N,))
    beta_ib = torch.normal(**to_tensor(dist['beta_ib']),size=(N,))
    delta_is = torch.normal(**to_tensor(dist['delta_is']),size=(N,))
    delta_ib = torch.normal(**to_tensor(dist['delta_ib']),size=(N,))
    # lm_i = np.exp(np.random.normal(**log_lm, size=(N, )))

    Z_j = MultivariateNormal(torch.tensor(dist['Z_j']['loc']), torch.eye(V)*dist['Z_j']['scale']).sample()  #(V, )
    tau_pass = MultivariateNormal(torch.tensor(dist['tau_pass']['loc']), torch.eye(V)*dist['tau_pass']['scale']).sample() #(V, )
    tau_shop = MultivariateNormal(torch.tensor(dist['tau_shop']['loc']), torch.eye(V)*dist['tau_shop']['scale']).sample() #(V, )
    
    graph = create_graph_and_product(config["graph_cfg_path"], V, K)
    product = graph.product_mat
    
    a_ik0 = MultivariateNormal(torch.tensor(dist['a_ik0']['loc']), torch.eye(K)*dist['a_ik0']['scale']).sample(torch.Size([N]))
    x_i0 = (torch.ones(N) * graph.outside).long()
    T_i0 = torch.zeros(N)

    a_model = AttractionModel(a_ik0, product, delta_is, delta_ib, w_v)
    v_model = VisitModel(x_i0, graph, dist['N_t']['rate'], Z_j, kappa, gamma_v)
    s_model = ShopModel(N, alpha_is, beta_is, w_s, gamma_s, tau_shop, tau_pass)
    b_model = BuyModel(N, product, alpha_ib, beta_ib, gamma_b, w_b)
    t_model = TimeModel(T_i0)
    mask = v_model.mask()

    for t in range(T_max - 1):
        noise_b = torch.normal(**to_tensor(dist['noise_b']), size=(N, K))
        noise_s = torch.normal(**to_tensor(dist['noise_s']), size=(N, ))
        noise_v = torch.normal(**to_tensor(dist['noise_v']), size=(N, V))

        v_model.visit(t)
        b_model.step(a_model.a_ikt, t_model.T_it, v_model.rho_jt, s_model.H_it, v_model.x_it, noise_b, mask)
        s_model.step(a_model.A_ijt, v_model.x_it, t_model.T_it, v_model.rho_jt, noise_s, mask)
        a_model.step(b_model.checkout, v_model.x_it, b_model.B_ikt, s_model.S_it, mask)
        v_model.step(a_model.A_ijt, noise_v)
        mask = v_model.mask()
        t_model.step(mask)
    v_model.visit(t)

    a_ikt_seq = torch.stack(a_model.log['a_ikt'])
    x_it_seq = torch.stack(v_model.log['x_it'])
    rho_jt_seq = torch.stack(v_model.log['rho_jt'])
    H_it_seq = torch.stack(s_model.log['H_it'])
    S_it_seq = torch.stack(s_model.log['S_it'])
    B_ikt_seq = torch.stack(b_model.log['B_ikt'])

    return dict(a_ikt=a_ikt_seq, x_it=x_it_seq, rho_jt=rho_jt_seq, H_it=H_it_seq, S_it=S_it_seq, B_ikt=B_ikt_seq, T_0=v_model.T_0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('-config', type=str, default="data/config/config.yaml", help="Path to configuration file")
    parser.add_argument('-save_path', type=str, default="data/dataset.h5", help="Path to save dataset")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    # convert to json type object
    config_json = json.dumps(config)

    # generate dataset
    train_data = generate_data(config)
    val_data = generate_data(config)

    save_h5py(args.save_path, config, bytes(config_json, 'utf-8'), train_data, val_data)

    """ Read data from HDF5 file
    h5f = h5py.File('data/dataset.h5', 'r')

    # Read the binary data from the dataset and convert it to a JSON object
    cfg = h5f['config']
    config = json.loads(cfg['config_file'][()])

    # Read the tensor data
    dataset = h5f['dataset']['a_ikt']
    a_ikt = torch.tensor(dataset[:]) # torch.Tensor object

    # Close the file
    h5f.close()
    """