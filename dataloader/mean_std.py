from dataloader import load_jetclass_label_as_tensor
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

labels=['QCD', 'Hbb', 'label_Hcc', 'label_Hgg', 'label_H4q',
                'label_Hqql', 'label_Zqq', 'label_Wqq', 'label_Tbqq', 'label_Tbl']
dict = {
    'label_QCD': [],
    'label_Hbb': [],
    'label_Hcc': [],
    'label_Hgg': [],
    'label_H4q': [],
    'label_Hqql': [],
    'label_Zqq': [],
    'label_Wqq': [],
    'label_Tbqq': [],
    'label_Tbl': []
}
for label in labels:
    dataloader = load_jetclass_label_as_tensor(label=label, start=10, end=90, batch_size=512)
    all_particles = []
    for x_particles, _, _ in dataloader:
        all_particles.append(x_particles)
    all_particles = torch.cat(all_particles, dim=0)  
    all_particles = all_particles.transpose(1, 2)       
    flat_particles = all_particles.reshape(-1, 3) 
    global_mean = flat_particles.mean(dim=0).to(device)
    global_std = flat_particles.std(dim=0).to(device) + 1e-6
    print(f"Global Mean for {label}: {global_mean}")
    print(f"Global Std for {label}: {global_std}")
    dict[label].append(global_mean)
    dict[label].append(global_std)
