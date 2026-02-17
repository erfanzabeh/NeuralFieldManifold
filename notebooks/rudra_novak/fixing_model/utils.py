import numpy as np
from sklearn.model_selection import train_test_split

def load_dataset_npz(path):
    d = np.load(path, allow_pickle=True)
    return {
        "X": d["X"],
        "A": d["A"],
        "p_true": d["p_true"],
        "class_id": d["class_id"],
        "class_names": d["class_names"],
        "noise_std": d["noise_std"],
        "meta": list(d["meta"]),
    }

def pinn_processor(npz_path, family=None, p_min=1, train_split=0.8, random_state=42):
    """
    Load a saved TVAR benchmark dataset and return stratified train/val arrays.
    
    Args
    ----
    npz_path : str
        Path to the .npz file saved by save_dataset_npz().
    family : str or None
        The class/family name to filter by. If None, use full dataset.
    p_min : int
        Minimum AR order used when generating the dataset. 
        Used to compute 0-indexed class labels for cross entropy.
    train_split : float
        Fraction of data to use for training (default 0.8 for 80/20 split).
    random_state : int
        Random seed for reproducibility.
    
    Returns
    -------
    X_train, X_val : arrays, shape [N_train, T], [N_val, T]
        Time series samples for train and validation.
    coeffs_train, coeffs_val : arrays, shape [N_train, T, p_max], [N_val, T, p_max]
        Time-varying AR coefficients for train and validation.
    p_train, p_val : arrays, shape [N_train], [N_val]
        True AR order for each sample, 0-indexed for cross entropy.
    class_id_train, class_id_val : arrays, shape [N_train], [N_val]
        Class/family ID for each sample.
    class_names : array
        Names of each class/family.
    """
    dataset = load_dataset_npz(npz_path)
    class_names = dataset["class_names"]

    # Use full dataset if no family specified, otherwise filter by family
    if family is None or family == '':
        X = dataset["X"][:]
        coeffs = dataset["A"][:]
        p_true_raw = dataset["p_true"][:]
        class_id = dataset["class_id"][:]
    else:
        datasets_by_class = {}
        for i, classname in enumerate(class_names):
            mask = dataset["class_id"] == i
            datasets_by_class[classname] = {
                "X": dataset["X"][mask],
                "A": dataset["A"][mask],
                "p_true": dataset["p_true"][mask],
                "class_id": dataset["class_id"][mask],
            }
        subdata = datasets_by_class[family]
        X = subdata["X"][:]
        coeffs = subdata["A"][:]
        p_true_raw = subdata["p_true"][:]
        class_id = subdata["class_id"][:]
    
    # Convert p_true to 0-indexed labels for cross entropy loss
    unique_p = np.sort(np.unique(p_true_raw))
    p_to_idx = {int(p): i for i, p in enumerate(unique_p)}
    p_true = np.array([p_to_idx[int(p)] for p in p_true_raw], dtype=np.int64)
    
    # Stratified train/val split to maintain balanced p distribution
    indices = np.arange(len(X))
    train_idx, val_idx = train_test_split(
        indices, 
        train_size=train_split, 
        stratify=p_true,  # Ensures balanced p values in both splits
        random_state=random_state
    )
    
    X_train, X_val = X[train_idx], X[val_idx]
    coeffs_train, coeffs_val = coeffs[train_idx], coeffs[val_idx]
    p_train, p_val = p_true[train_idx], p_true[val_idx]
    class_id_train, class_id_val = class_id[train_idx], class_id[val_idx]
    
    # Print distribution info
    print(f"Total samples: {len(X)} | Train: {len(X_train)} | Val: {len(X_val)}")
    print(f"Train p distribution: {dict(zip(*np.unique(p_train, return_counts=True)))}")
    print(f"Val p distribution:   {dict(zip(*np.unique(p_val, return_counts=True)))}")
    
    return X_train, coeffs_train, p_train, class_id_train, X_val, coeffs_val, p_val, class_id_val, class_names