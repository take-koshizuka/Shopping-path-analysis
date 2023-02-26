from util import load_h5py
from torch.utils.data import Dataset, DataLoader

class SimulationDataset(Dataset):
    def __init__(self, data):
        self.x_it = data['x_it']
        self.a_ikt = data['a_ikt']
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        
        return {'data': x, 'target': y}
    
    def __len__(self):
        return len(self.data)

def setup_data_loaders(config, train_data, val_data):
    tr_ds = SimulationDataset(train_data)
    va_ds = SimulationDataset(val_data)
    tr_dl = DataLoader(tr_ds, batch_size=config['bs'], shuffle=True, drop_last=True)
    va_dl = DataLoader(va_ds, batch_size=config['bs'], shuffle=False)
    return tr_dl, va_dl
