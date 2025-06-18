import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import awkward as ak
import uproot
import vector
vector.register_awkward()

def _pad_with_mask(a, maxlen, value=0, dtype='float32'):
    if isinstance(a, ak.Array):
        lengths = ak.num(a, axis=1)
        padded = ak.pad_none(a, maxlen, clip=True)
        mask = ak.local_index(padded) < lengths[:, np.newaxis]

        filled = ak.fill_none(padded, value)
        final = ak.to_numpy(ak.values_astype(filled, dtype))
        mask_np = ak.to_numpy(mask).astype('float32')

        print(f"Padded array shape: {final.shape}")
        print(f"Mask shape: {mask_np.shape}")
        return final, mask_np

    else:
        batch_size = len(a)
        x = np.full((batch_size, maxlen) + np.shape(a[0][0]), value, dtype=dtype)
        mask = np.zeros((batch_size, maxlen), dtype='float32')

        for idx, jet in enumerate(a):
            n = min(len(jet), maxlen)
            x[idx, :n] = jet[:n]
            mask[idx, :n] = 1.0

        return x, mask

def read_file(
        filepath,
        max_num_particles=128,
        particle_features=['part_pt', 'part_eta', 'part_phi'],
        jet_features=['jet_pt', 'jet_eta', 'jet_phi'],
        labels=['label_QCD', 'label_Hbb', 'label_Hcc', 'label_Hgg', 'label_H4q',
                'label_Hqql', 'label_Zqq', 'label_Wqq', 'label_Tbqq', 'label_Tbl']):

    table = uproot.open(filepath)['tree'].arrays()

    p4 = vector.zip({'px': table['part_px'],
                     'py': table['part_py'],
                     'pz': table['part_pz'],
                     'energy': table['part_energy']})
    table['part_pt'] = p4.pt
    table['part_eta'] = p4.eta
    table['part_phi'] = p4.phi

    padded_features = []
    masks = []
    for n in particle_features:
        padded, mask = _pad_with_mask(table[n], maxlen=max_num_particles)
        padded_features.append(padded)
        masks.append(mask)

    x_particles = np.stack(padded_features, axis=1)  # Shape: (N, F, T)
    mask = masks[0]  # Assumes all particle features have the same mask

    x_jets = np.stack([ak.to_numpy(table[n]).astype('float32') for n in jet_features], axis=1)
    y = np.stack([ak.to_numpy(table[n]).astype('int') for n in labels], axis=1)

    return x_particles, x_jets, y, mask

def load_jetclass_label_as_tensor(label="HToBB", start=10, end=15, batch_size=512):
    x_particles_list = []
    x_jet_list = []
    y_list = []
    mask_list = []

    for i in range(start, end):
        print(f"Loading file {i} for label {label}")
        file_path = f"/pscratch/sd/h/haoming/particle_transformer/dataset/JetClass/{label}_{i:03d}.root"
        if not os.path.exists(file_path):
            continue
        try:
            x_particles, x_jet, y, mask = read_file(file_path)
            x_particles_list.append(x_particles)
            x_jet_list.append(x_jet)
            y_list.append(y)
            mask_list.append(mask)
        except Exception as e:
            continue

    x_particles_all = np.concatenate(x_particles_list, axis=0)
    x_jet_all = np.concatenate(x_jet_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    mask_all = np.concatenate(mask_list, axis=0)

    x_particles_tensor = torch.tensor(x_particles_all, dtype=torch.float32)
    x_jet_tensor = torch.tensor(x_jet_all, dtype=torch.float32)
    y_tensor = torch.tensor(y_all, dtype=torch.long)
    mask_tensor = torch.tensor(mask_all, dtype=torch.float32)

    dataset = TensorDataset(x_particles_tensor, x_jet_tensor, y_tensor, mask_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

def load_jetclass_label_as_dataset(label="HToBB", start=10, end=15):
    x_particles_list = []
    x_jet_list = []
    y_list = []
    mask_list = []

    for i in range(start, end):
        print(f"Loading file {i} for label {label}")
        file_path = f"/pscratch/sd/h/haoming/particle_transformer/dataset/JetClass/{label}_{i:03d}.root"
        if not os.path.exists(file_path):
            continue
        try:
            x_particles, x_jet, y, mask = read_file(file_path)
            x_particles_list.append(x_particles)
            x_jet_list.append(x_jet)
            y_list.append(y)
            mask_list.append(mask)
        except Exception as e:
            print(f"\u26a0\ufe0f Skipped file {file_path}: {e}")
            continue

    if not x_particles_list:
        raise RuntimeError("No valid files loaded. Dataset is empty.")

    x_particles_all = np.concatenate(x_particles_list, axis=0)
    x_jet_all = np.concatenate(x_jet_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    mask_all = np.concatenate(mask_list, axis=0)

    x_particles_tensor = torch.tensor(x_particles_all, dtype=torch.float32)
    x_jet_tensor = torch.tensor(x_jet_all, dtype=torch.float32)
    y_tensor = torch.tensor(y_all, dtype=torch.long)
    mask_tensor = torch.tensor(mask_all, dtype=torch.float32)

    dataset = TensorDataset(x_particles_tensor, x_jet_tensor, y_tensor, mask_tensor)
    return dataset

# Test loading one batch
dataloader = load_jetclass_label_as_tensor(label="HToBB", start=10, end=11, batch_size=512)
x_particles, x_jets, y, mask = next(iter(dataloader))

