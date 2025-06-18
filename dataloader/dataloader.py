import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import awkward as ak
import uproot
import vector
vector.register_awkward()


def read_file(
        filepath,
        max_num_particles=128,
        particle_features=['part_pt', 'part_eta', 'part_phi'],
        jet_features=['jet_pt', 'jet_eta', 'jet_phi'],
        labels=['label_QCD', 'label_Hbb', 'label_Hcc', 'label_Hgg', 'label_H4q',
                'label_Hqql', 'label_Zqq', 'label_Wqq', 'label_Tbqq', 'label_Tbl']):
    """Loads a single file from the JetClass dataset.

    **Arguments**

    - **filepath** : _str_
        - Path to the ROOT data file.
    - **max_num_particles** : _int_
        - The maximum number of particles to load for each jet. 
        Jets with fewer particles will be zero-padded, 
        and jets with more particles will be truncated.
    - **particle_features** : _List[str]_
        - A list of particle-level features to be loaded. 
        The available particle-level features are:
            - part_px
            - part_py
            - part_pz
            - part_energy
            - part_pt
            - part_eta
            - part_phi
            - part_deta: np.where(jet_eta>0, part_eta-jet_p4, -(part_eta-jet_p4))
            - part_dphi: delta_phi(part_phi, jet_phi)
            - part_d0val
            - part_d0err
            - part_dzval
            - part_dzerr
            - part_charge
            - part_isChargedHadron
            - part_isNeutralHadron
            - part_isPhoton
            - part_isElectron
            - part_isMuon
    - **jet_features** : _List[str]_
        - A list of jet-level features to be loaded. 
        The available jet-level features are:
            - jet_pt
            - jet_eta
            - jet_phi
            - jet_energy
            - jet_nparticles
            - jet_sdmass
            - jet_tau1
            - jet_tau2
            - jet_tau3
            - jet_tau4
    - **labels** : _List[str]_
        - A list of truth labels to be loaded. 
        The available label names are:
            - label_QCD
            - label_Hbb
            - label_Hcc
            - label_Hgg
            - label_H4q
            - label_Hqql
            - label_Zqq
            - label_Wqq
            - label_Tbqq
            - label_Tbl

    **Returns**

    - x_particles(_3-d numpy.ndarray_), x_jets(_2-d numpy.ndarray_), y(_2-d numpy.ndarray_)
        - `x_particles`: a zero-padded numpy array of particle-level features 
                         in the shape `(num_jets, num_particle_features, max_num_particles)`.
        - `x_jets`: a numpy array of jet-level features
                    in the shape `(num_jets, num_jet_features)`.
        - `y`: a one-hot encoded numpy array of the truth lables
               in the shape `(num_jets, num_classes)`.
    """

    def _pad(a, maxlen, value=0, dtype='float32'):
        if isinstance(a, np.ndarray) and a.ndim >= 2 and a.shape[1] == maxlen:
            return a
        elif isinstance(a, ak.Array):
            if a.ndim == 1:
                a = ak.unflatten(a, 1)
            a = ak.fill_none(ak.pad_none(a, maxlen, clip=True), value)
            return ak.values_astype(a, dtype)
        else:
            x = (np.ones((len(a), maxlen)) * value).astype(dtype)
            for idx, s in enumerate(a):
                if not len(s):
                    continue
                trunc = s[:maxlen].astype(dtype)
                x[idx, :len(trunc)] = trunc
            return x

    table = uproot.open(filepath)['tree'].arrays()

    p4 = vector.zip({'px': table['part_px'],
                     'py': table['part_py'],
                     'pz': table['part_pz'],
                     'energy': table['part_energy']})
    table['part_pt'] = p4.pt
    table['part_eta'] = p4.eta
    table['part_phi'] = p4.phi

    x_particles = np.stack([ak.to_numpy(_pad(table[n], maxlen=max_num_particles)) for n in particle_features], axis=1)
    x_jets = np.stack([ak.to_numpy(table[n]).astype('float32') for n in jet_features], axis=1)
    y = np.stack([ak.to_numpy(table[n]).astype('int') for n in labels], axis=1)

    return x_particles, x_jets, y

def load_jetclass_label_as_tensor(label="HToBB", start=10, end=15, batch_size=512):
    x_particles_list = []
    x_jet_list = []
    y_list = []

    for i in range(start, end):
        print(f"Loading file {i} for label {label}")
        file_path = f"/pscratch/sd/h/haoming/particle_transformer/dataset/JetClass/{label}_{i:03d}.root"
        if not os.path.exists(file_path):
            continue
        try:
            x_particles, x_jet, y = read_file(file_path)
            x_particles_list.append(x_particles)
            x_jet_list.append(x_jet)
            y_list.append(y)
        except Exception as e:
            print(f"⚠️ Skipped file {file_path}: {e}")
            continue

    # Concatenate across all files
    x_particles_all = np.concatenate(x_particles_list, axis=0)
    x_jet_all = np.concatenate(x_jet_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)

    # Convert to torch tensors
    x_particles_tensor = torch.tensor(x_particles_all, dtype=torch.float32)
    x_jet_tensor = torch.tensor(x_jet_all, dtype=torch.float32)
    y_tensor = torch.tensor(y_all, dtype=torch.long)

    # Build dataset and dataloader
    dataset = TensorDataset(x_particles_tensor, x_jet_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

def load_jetclass_label_as_dataset(label="HToBB", start=10, end=15):
    x_particles_list = []
    x_jet_list = []
    y_list = []

    for i in range(start, end):
        print(f"Loading file {i} for label {label}")
        file_path = f"/pscratch/sd/h/haoming/particle_transformer/dataset/JetClass/{label}_{i:03d}.root"
        if not os.path.exists(file_path):
            continue
        try:
            x_particles, x_jet, y = read_file(file_path)
            x_particles_list.append(x_particles)
            x_jet_list.append(x_jet)
            y_list.append(y)
        except Exception as e:
            print(f"⚠️ Skipped file {file_path}: {e}")
            continue

    if not x_particles_list:
        raise RuntimeError("No valid files loaded. Dataset is empty.")

    x_particles_all = np.concatenate(x_particles_list, axis=0)
    x_jet_all = np.concatenate(x_jet_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)

    x_particles_tensor = torch.tensor(x_particles_all, dtype=torch.float32)
    x_jet_tensor = torch.tensor(x_jet_all, dtype=torch.float32)
    y_tensor = torch.tensor(y_all, dtype=torch.long)

    dataset = TensorDataset(x_particles_tensor, x_jet_tensor, y_tensor)
    return dataset



