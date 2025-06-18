import os
import glob
import numpy as np
from omegaconf import OmegaConf
from dataloader import read_file  # your working `read_file()` function

def load_class_from_yaml(yaml_path, class_key="HToBB", mode="train"):
    """
    Loads all ROOT files for a given class (like HToBB) using config.yaml.
    
    Args:
        yaml_path (str): Path to your YAML file.
        class_key (str): The physics class to load, e.g. "HToBB".
        mode (str): One of "train", "val", or "test".

    Returns:
        x_particles, x_jets, y: All concatenated arrays.
    """
    # Load YAML config
    cfg = OmegaConf.load(yaml_path)

    # Resolve paths
    mode_key = f"dataset_kwargs_{mode}"
    files_dict = cfg[mode_key]["files_dict"]

    # Find matching file paths for the selected class
    all_globs = []
    for entry in files_dict.get(class_key, []):
        resolved = entry.replace("${data.data_dir}", cfg["data_dir"])
        all_globs.extend(glob.glob(resolved))

    if not all_globs:
        raise FileNotFoundError(f"No files found for class {class_key} with glob {entry}")

    print(f"Found {len(all_globs)} files for {class_key}")

    # Prepare lists to store data
    x_parts_list, x_jets_list, y_list = [], [], []

    # Load each file using your existing function
    for filepath in all_globs:
        x_parts, x_jet, y = read_file(
            filepath,
            max_num_particles=cfg["dataset_kwargs_common"]["pad_length"],
            particle_features=list(cfg["dataset_kwargs_common"]["feature_dict"].keys()),
            labels=cfg["dataset_kwargs_common"]["labels_to_load"],
        )
        x_parts_list.append(x_parts)
        x_jets_list.append(x_jet)
        y_list.append(y)

    # Concatenate all files into one big array
    return (
        np.concatenate(x_parts_list, axis=0),
        np.concatenate(x_jets_list, axis=0),
        np.concatenate(y_list, axis=0),
    )
yaml_path = "/pscratch/sd/h/haoming/Projects/hep_models/dataloader/config.yaml"
x_particles, x_jets, y = load_class_from_yaml(yaml_path, class_key="HToBB", mode="train")

print(x_particles.shape)  # (total_jets, num_features, max_particles)
print(x_jets.shape)       # (total_jets, num_jet_features)
print(y.shape)            # (total_jets, num_classes)