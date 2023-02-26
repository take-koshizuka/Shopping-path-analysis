import csv
import json
import torch
import h5py
from collections import OrderedDict

def to_tensor(rv: dict):
    """
        Convert scalar-type parameters to a torch.Tensor.
        rv: dictionary (param name) -> (param value)
    """
    for name, param in rv.items():
        if isinstance(param, dict):
            rv[name] = to_tensor(param)

        if isinstance(param, int) or isinstance(param, float) or isinstance(param, list):
            rv[name] = torch.tensor(param)
    return rv

class StoreGraph:
    def __init__(self, V: int, K:int, edges: dict, product: dict, outside_state: str="OUTSIDE", init_state: str="INIT", checkout_state: str="CHECKOUT") -> None:
        self.V = V
        self.K = K
        nodes = edges.keys()
        self.node2idx = OrderedDict({node: i for i, node in enumerate(nodes)})
        self.adj = torch.zeros(self.V, self.V).long()
        self.product_mat = torch.zeros(self.V, self.K)
        for vi in nodes:
            vi_idx = self.node2idx[vi]
            for vj in edges[vi]:
                vj_idx = self.node2idx[vj]
                self.adj[vi_idx, vj_idx] = 1
            for pj in product[vi]:
                # pj_idx = self.product2idx[pj]
                pj_idx = pj
                self.product_mat[vi_idx, pj_idx] = 1

        self.outside_state = outside_state
        self.init_state = init_state
        self.checkout_state = checkout_state

    @property
    def outside(self):
        return self.node2idx[self.outside_state]
    
    @property
    def init(self):
        return self.node2idx[self.init_state]

    @property
    def checkout(self):
        return self.node2idx[self.checkout_state]
    
def create_graph_and_product(cfg_path, V, K):
    edges, product = {}, {}
    with open(cfg_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            edges[row[0]] = row[1].split('-')
            p = row[2].split('-')
            if p[0] == '': 
                product[row[0]] = []
            else: 
                product[row[0]] = list(map(int, p))
    g = StoreGraph(V, K, edges, product)
    return g


def save_h5py(save_path, config, config_bytes, train_data, val_data):
    # save dataset
    h5f = h5py.File(save_path, 'w')

    group_config = h5f.create_group('config')
    group_config.create_dataset('config_file', data=config_bytes)

    group_data = h5f.create_group('train_data')
    group_data.create_dataset('a_ikt', (config['T_max'], config['N'], config['K']), data=train_data['a_ikt'], dtype=float)
    group_data.create_dataset('x_it', (config['T_max'], config['N']), data=train_data['x_it'], dtype=int)
    group_data.create_dataset('rho_jt', (config['T_max'], config['V']), data=train_data['rho_jt'], dtype=float)
    group_data.create_dataset('T_0', (config['N'],), data=train_data['T_0'], dtype=int)
    group_data.create_dataset('H_it', (config['T_max'], config['N']), data=train_data['H_it'], dtype=int)
    group_data.create_dataset('S_it', (config['T_max'], config['N']), data=train_data['S_it'], dtype=int)
    group_data.create_dataset('B_ikt', (config['T_max'], config['N'], config['K']), data=train_data['B_ikt'], dtype=int)

    group_data = h5f.create_group('val_data')
    group_data.create_dataset('a_ikt', (config['T_max'], config['N'], config['K']), data=val_data['a_ikt'], dtype=float)
    group_data.create_dataset('x_it', (config['T_max'], config['N']), data=val_data['x_it'], dtype=int)
    group_data.create_dataset('rho_jt', (config['T_max'], config['V']), data=val_data['rho_jt'], dtype=float)
    group_data.create_dataset('T_0', (config['N'],), data=val_data['T_0'], dtype=int)
    group_data.create_dataset('H_it', (config['T_max'], config['N']), data=val_data['H_it'], dtype=int)
    group_data.create_dataset('S_it', (config['T_max'], config['N']), data=val_data['S_it'], dtype=int)
    group_data.create_dataset('B_ikt', (config['T_max'], config['N'], config['K']), data=val_data['B_ikt'], dtype=int)

    h5f.close()

def load_h5py(load_path):
    h5f = h5py.File(load_path, 'r')

    # Read the binary data from the dataset and convert it to a JSON object
    cfg = h5f['config']
    config = json.loads(cfg['config_file'][()])
    # Read the tensor data
    # dataset = h5f['dataset']['a_ikt']
    dataset_names = ["x_it", "a_ikt", "rho_jt", "T_0", "H_it", "S_it", "B_ikt"]
    train_data = { name: torch.tensor(h5f['train_data'][name][:]) for name in dataset_names }
    val_data = { name: torch.tensor(h5f['val_data'][name][:]) for name in dataset_names }
    # Close the file
    h5f.close()

    return dict(config=config, train_data=train_data, val_data=val_data)