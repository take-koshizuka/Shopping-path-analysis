import argparse
import yaml
import pyro
import json
import pyro.distributions as dist
import pyro.contrib.examples.util  # patches torchvision
from pyro.infer import SVI, Trace_ELBO, JitTrace_ELBO
from pyro.optim import Adam
from pyro.infer.mcmc.api import MCMC
from pyro.infer.mcmc import NUTS
from models.baseline import BSAV_Model
from util import load_h5py
from data import setup_data_loaders
from pyro.poutine import trace

def train_mcmc(config):
    # clear param store
    pyro.clear_param_store()
    # train_loader, test_loader
    D = load_h5py(config['dataset_path'])
    config.update(D['config'])

    # setup the VAE
    model = BSAV_Model(config)



    pyro.set_rng_seed(2)
    kernel = NUTS(model.model)
    mcmc = MCMC(kernel, num_samples=100, warmup_steps=50)

    train_data = D['train_data']
    train_data.pop('a_ikt')
    train_data.pop('H_it')
    mcmc.run(**train_data)

    print(mcmc.summary())

def train_svi(config):
    # train_loader, test_loader
    tr_dl, te_dl = setup_data_loaders(config['dataset_path'])

    # setup the VAE
    model = BSAV_Model(config)

    # setup the optimizer
    adam_args = {"lr": config['lr']}
    optimizer = Adam(adam_args)

    # setup the inference algorithm
    elbo = JitTrace_ELBO() if config['jit'] else Trace_ELBO()
    svi = SVI(model.model, model.guide, optimizer, loss=elbo)

    train_elbo = []
    test_elbo = []
    # training loop
    for epoch in range(args.num_epochs):
        # initialize loss accumulator
        epoch_loss = 0.0
        # do a training epoch over each mini-batch x returned
        # by the data loader
        for x, _ in tr_dl:
            # if on GPU put mini-batch into CUDA memory
            if args.cuda:
                x = x.cuda()
            # do ELBO gradient and accumulate loss
            epoch_loss += svi.step(x)

        # report training diagnostics
        normalizer_train = len(tr_dl.dataset)
        total_epoch_loss_train = epoch_loss / normalizer_train
        train_elbo.append(total_epoch_loss_train)
        print(
            "[epoch %03d]  average training loss: %.4f"
            % (epoch, total_epoch_loss_train)
        )

        if epoch % args.test_frequency == 0:
            # initialize loss accumulator
            test_loss = 0.0
            # compute the loss over the entire test set
            for i, (x, _) in enumerate(te_dl):
                # if on GPU put mini-batch into CUDA memory
                if args.cuda:
                    x = x.cuda()
                # compute ELBO estimate and accumulate loss
                test_loss += svi.evaluate_loss(x)

            # report test diagnostics
            normalizer_test = len(te_dl.dataset)
            total_epoch_loss_test = test_loss / normalizer_test
            test_elbo.append(total_epoch_loss_test)
            print(
                "[epoch %03d]  average test loss: %.4f" % (epoch, total_epoch_loss_test)
            )

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('-alg', type=str, default="svi", choices=['svi', 'mcmc'])
    parser.add_argument('-config', type=str, default="config/train.yaml", help="Path to configuration file")
    parser.add_argument('-dataset_path', type=str, default=None, help="Path to save dataset")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if not args.dataset_path is None:
        config['dataset_path'] = args.dataset_path
    
    if args.alg == 'svi':
        train_svi(config)
    
    elif args.alg == 'mcmc':
        train_mcmc(config)