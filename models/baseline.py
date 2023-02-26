import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.log_normal import LogNormal
from torch.distributions.categorical import Categorical
from torch.distributions.geometric import Geometric
import pyro
import pyro.distributions as dist
import pyro.contrib.examples.util  # patches torchvision
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from util import create_graph_and_product, to_tensor

class BSAV_Model(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.N = config['N']
        self.K = config['K']
        self.V = config['V']
        self.prior = to_tensor(config['model']['prior'])
        self.noise = to_tensor(config['model']['noise'])
        self.T_max = config['T_max']

        self.graph = create_graph_and_product(config["graph_cfg_path"], self.V, self.K)
        self.product = self.graph.product_mat

    def model(self, x_it, S_it, B_ikt, T_0, rho_jt):
        B = x_it.size(1) # Batch size
        with pyro.plate("a_ik0_plate", B):
            a_ik0 = pyro.sample("a_ik0", 
                dist.MultivariateNormal(self.prior['a_ik0']['mean']*torch.ones(self.K), self.prior['a_ik0']['std']*torch.eye(self.K)))

        w_v = pyro.sample('w_v', dist.Normal(self.prior['w_v']['mean'], self.prior['w_v']['std']))
        w_s = pyro.sample('w_s', dist.Normal(self.prior['w_s']['mean'], self.prior['w_s']['std']))
        w_b = pyro.sample('w_b', dist.Normal(self.prior['w_b']['mean'], self.prior['w_b']['std']))
        gamma_v = pyro.sample('gamma_v', dist.Normal(self.prior['gamma_v']['mean'], self.prior['gamma_v']['std']))
        gamma_s = pyro.sample('gamma_s', dist.Normal(self.prior['gamma_s']['mean'], self.prior['gamma_s']['std']))
        gamma_b = pyro.sample('gamma_b', dist.Normal(self.prior['gamma_b']['mean'], self.prior['gamma_b']['std']))

        # sampling parameters
        #with pyro.plate("parameters_plate", B):
        kappa = torch.exp(pyro.sample('kappa', dist.Normal(self.prior['kappa']['mean'], self.prior['kappa']['std'])).expand([B, ]))
        alpha_is = pyro.sample('alpha_is', dist.Normal(self.prior['alpha_is']['mean'], self.prior['alpha_is']['std']))
        alpha_ib = pyro.sample('alpha_ib', dist.Normal(self.prior['alpha_ib']['mean'], self.prior['alpha_ib']['std']))
        beta_is = pyro.sample('beta_is', dist.Normal(self.prior['beta_is']['mean'], self.prior['beta_is']['std']))
        beta_ib = pyro.sample('beta_ib', dist.Normal(self.prior['beta_ib']['mean'], self.prior['beta_ib']['std']))
        delta_is = pyro.sample('delta_is', dist.Normal(self.prior['delta_is']['mean'], self.prior['delta_is']['std'])).expand([B, ])
        delta_ib = pyro.sample('delta_ib', dist.Normal(self.prior['delta_ib']['mean'], self.prior['delta_ib']['std'])).expand([B, ])
        # lm_i = np.exp(np.random.normal(**log_lm, size=(N, )))

        Z_j = pyro.sample('Z_j', dist.Normal(self.prior['Z_j']['mean'], self.prior['Z_j']['std']).expand([self.V, ])) #(V, )
        tau_pass = pyro.sample('tau_pass', dist.Normal(self.prior['tau_pass']['mean'], self.prior['tau_pass']['std']).expand([self.V, ])) #(V, )
        tau_shop = pyro.sample('tau_shop', dist.Normal(self.prior['tau_shop']['mean'], self.prior['tau_shop']['std']).expand([self.V, ])) #(V, )
        
        # initial value from data
        x_i0 = torch.diag(x_it[T_0], 0)
        #S_i0 = torch.diag(S_it[T_0], 0) #(B,)
        #B_ik0 = B_ikt[T_0][torch.arange(B), torch.arange(B), :] #(B, K)
        T_i0 = torch.zeros(B)
        rho_jt = rho_jt[T_0] #(B,V)

        a_model = AttractionModel(a_ik0, self.product, delta_is, delta_ib, w_v)
        v_model = VisitModel(x_i0, rho_jt, self.graph, Z_j, kappa, gamma_v)
        s_model = ShopModel(B, alpha_is, beta_is, w_s, gamma_s, tau_shop, tau_pass)
        b_model = BuyModel(B, self.product, alpha_ib, beta_ib, gamma_b, w_b)
        t_model = TimeModel(T_i0)
        mask = v_model.mask()
        
        for t in range(self.T_max - 1):
            noise_b = torch.normal(**self.noise['noise_b'], size=(B, self.K))
            noise_s = torch.normal(**self.noise['noise_s'], size=(B, ))
            noise_v = torch.normal(**self.noise['noise_v'], size=(B, self.V))

            ti = torch.clamp(T_0 + t, max=self.T_max-1)
            b_model.step(t, a_model.a_ikt, t_model.T_it, v_model.rho_jt, s_model.H_it, v_model.x_it, noise_b, mask, obs=B_ikt[ti][torch.arange(B), torch.arange(B), :])
            s_model.step(t, a_model.A_ijt, v_model.x_it, t_model.T_it, v_model.rho_jt, noise_s, mask, obs=torch.diag(S_it[ti], 0))
            a_model.step(b_model.checkout, v_model.x_it, b_model.B_ikt, s_model.S_it, mask)
            a_model.step(torch.zeros(B, self.K), v_model.x_it, torch.zeros(B, self.K), torch.zeros(B), mask)
            v_model.step(t, a_model.A_ijt, noise_v, obs=torch.diag(x_it[ti], 0))
            mask = v_model.mask()
            t_model.step(mask)
    

class MaskedSoftmax(nn.Module):
    def __init__(self):
        super(MaskedSoftmax, self).__init__()
        self.softmax = nn.Softmax(1)

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
        self.a_ikt = a_ikt.double() #(N, K)
        self.N = a_ikt.size(0)
        self.K = a_ikt.size(1)
        self.product = product.double() #(V, K)
        self.delta_is = delta_is.repeat(self.K, 1).T
        self.delta_ib = delta_ib.repeat(self.K, 1).T
        self.w_v = w_v
        self.V = self.product.size(0)
        assert self.product.size(1) == self.K
        self.A_ijt = self.cal_zone_attraction(self.a_ikt) # (N, V)
        # self.G_ijt = 
    def I(self, x):
        return torch.stack([ self.product[v] for v in x ]) #(N,K)
    
    def cal_zone_attraction(self, a_ikt):
        return torch.log(torch.matmul(torch.exp(a_ikt), self.product.T)+1.0)

    def step(self, checkout, x_it, B_ikt, S_it, mask):
        # checkout: (N, K)
        mask_checkout = torch.logical_and(checkout, mask.repeat(self.K, 1).T)
        a_ikt_n = torch.where(mask_checkout, 
                self.a_ikt + self.delta_ib * B_ikt + self.delta_is* self.I(x_it),
                self.a_ikt + self.w_v * S_it.repeat(self.K, 1).T,
        )
        a_ikt_n = torch.where(mask.repeat(self.K, 1).T, a_ikt_n, self.a_ikt)
        self.A_ijt = self.cal_zone_attraction(a_ikt_n) # (N,V)
        # self.G_ijt = 
        self.a_ikt = a_ikt_n

class VisitModel:
    def __init__(self, x_it, rho_jt, graph, Z_j, kappa, gamma_v) -> None:
        self.x_it = x_it #(N, )
        self.rho_jt = rho_jt #(N, V)
        self.N = self.x_it.size(0)
        self.graph = graph
        self.V = graph.V
        self.acc_N = 0
        self.Z_j = Z_j.repeat(self.N, 1)
        self.kappa = kappa.repeat(self.V, 1).T
        self.gamma_v = gamma_v
        self.T_0 = torch.zeros(self.N)
        self.softmax = MaskedSoftmax()

    def mask(self):
        return torch.logical_and(self.x_it != self.graph.outside, self.x_it != self.graph.checkout)
    
    def step(self, t, G_ijt, noise_v, obs):
        u_v = self.Z_j + self.kappa * G_ijt + self.gamma_v * self.rho_jt + noise_v #(N, V)
        mask_v = self.graph.adj[self.x_it] # (N, V)
        p = self.softmax(u_v, mask=mask_v)
        with pyro.plate(f"data_plate_{t}", self.N):
            x_it_n = pyro.sample(f"x_{t}", dist.Categorical(p), obs=obs)
        self.x_it = x_it_n

class ShopModel:
    def __init__(self, N, alpha_is, beta_is, w_s, gamma_s, tau_shop, tau_pass) -> None:
        self.N = N
        self.H_it = torch.zeros(self.N) #(N,)
        self.S_it = torch.zeros(self.N)

        self.alpha_is = alpha_is
        self.beta_is = beta_is
        self.w_s = w_s
        self.gamma_s = gamma_s
        self.tau = torch.stack([tau_pass, tau_shop])
    
    def step(self, t, A_ijt, x_it, T_it, rho_jt, noise_s, mask, obs):
        u_s = self.alpha_is + self.beta_is * A_ijt[torch.arange(self.N), x_it] + \
                self.w_s * T_it + self.gamma_s * rho_jt[torch.arange(self.N), x_it] + noise_s # (N, )
        
        #with pyro.plate(f"H_plate_{t}", self.N):
        #    H_it = pyro.sample(f"H_i{t}", dist.Binomial(1, logits=u_s))
        # print(u_s.size(), H_it.size())
        H_it = torch.bernoulli(torch.sigmoid(u_s))
        H_it = torch.where(mask, int(0), H_it).long()
        logits = self.tau[H_it][..., x_it][..., torch.arange(self.N), torch.arange(self.N)]
        #print(logits.size())
        with pyro.plate(f"S_plate_{t}", self.N):
            S_it = pyro.sample(f"S_i{t}", dist.Geometric(logits=logits), obs=obs.long()).long()
        S_it = torch.where(mask, 0, S_it)
        
        self.H_it = H_it
        self.S_it = S_it
        return self.H_it, self.S_it

class BuyModel:
    def __init__(self, N, product, alpha_ib, beta_ib, gamma_b, w_b) -> None:
        self.N = N
        self.product = product #(V, K)
        self.K = product.size(1)
        self.B_ikt = torch.zeros(self.N, self.K) #(N, K)
        self.checkout = torch.zeros(self.N, self.K) #(N, K)

        self.alpha_ib = alpha_ib.repeat(self.K, 1).T
        self.beta_ib = beta_ib.repeat(self.K, 1).T
        self.gamma_b = gamma_b
        self.w_b = w_b

    def I(self, x):
        return torch.stack([ self.product[v] for v in x ])

    def step(self, t, a_ikt, T_it, rho_jt, H_it, x_it, noise_b, mask, obs):
        # print(T_it.repeat(self.K, 1).T.size(), self.beta_ib.size(), self.alpha_ib.size(), rho_jt[x_it].repeat(self.K, 1).T.size())
        u_b = self.alpha_ib + self.beta_ib * a_ikt + self.w_b * T_it.repeat(self.K, 1).T + \
                    self.gamma_b * rho_jt[torch.arange(self.N), x_it].repeat(self.K, 1).T + noise_b #(N, K)
            
        with pyro.plate(f"B_plate_{t}", self.N, dim=-2), pyro.plate(f"B_plate_k{t}", self.K, dim=-1):
            B_ikt = pyro.sample(f"B_ik{t}", dist.Binomial(1, logits=u_b), obs=obs.long()).long()

        mask_i = (H_it == 1).repeat(self.K, 1).T
        B_ikt = torch.where(mask_i, 0., B_ikt)
        B_ikt = torch.where(self.I(x_it).bool(), 0., B_ikt)
        B_ikt = torch.where(mask.repeat(self.K, 1).T, 0., B_ikt)
        self.B_ikt = B_ikt.long()
        self.checkout = torch.logical_or(self.checkout, self.B_ikt)
        return self.B_ikt

class TimeModel:
    def __init__(self, T_it):
        self.T_it = T_it

    def step(self, mask):
        self.T_it[mask] = self.T_it[mask] + 1.