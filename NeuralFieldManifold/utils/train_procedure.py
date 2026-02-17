from tqdm import tqdm
import torch
import numpy as np

from .losses import loss_p, loss_ar, loss_energy, loss_smooth, loss_order


def train_loop(
    model,
    train_loader,
    val_loader,
    n_epochs=100,
    lr=1e-3,
    lambda_p=10.0,
    lambda_ar=1.0,
    lambda_energy=0.1,
    lambda_smooth=0.05,
    lambda_order=0.0,
    P0=0.5,
    W=40,
    p_max=6,
    device='cuda'
):
    """Train a model with the composite PINN-style loss.

    Optimises the weighted sum of five loss terms (classification,
    AR reconstruction, energy constraint, smoothness, and order
    regularisation) using Adam with gradient clipping and
    ``ReduceLROnPlateau`` scheduling.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train.  Must follow the ``(coeffs, p_logits, p_hard,
        x_hat)`` forward convention.
    train_loader : DataLoader
        Training data yielding ``(x_batch, p_batch)`` tuples.
    val_loader : DataLoader
        Validation data yielding ``(x_batch, p_batch)`` tuples.
    n_epochs : int, optional
        Number of training epochs. Default is 100.
    lr : float, optional
        Initial learning rate for Adam. Default is 1e-3.
    lambda_p : float, optional
        Weight for the AR-order cross-entropy loss. Default is 10.0.
    lambda_ar : float, optional
        Weight for the signal reconstruction MSE loss. Default is 1.0.
    lambda_energy : float, optional
        Weight for the energy constraint loss. Default is 0.1.
    lambda_smooth : float, optional
        Weight for the coefficient smoothness loss. Default is 0.05.
    lambda_order : float, optional
        Weight for the order regulariser. Default is 0.0 (disabled).
    P0 : float, optional
        Target power level for the energy loss. Default is 0.5.
    W : int, optional
        Sliding-window size for the energy loss. Default is 40.
    p_max : int, optional
        Maximum AR order (controls how many leading time steps are
        excluded in the AR loss). Default is 6.
    device : str or torch.device, optional
        Compute device. Default is ``'cuda'``.

    Returns
    -------
    history : dict
        Dictionary mapping metric names (e.g. ``'train_loss'``,
        ``'val_p_acc'``) to lists of per-epoch values.
    """
    params = list(model.parameters())
    
    if len(params) > 0:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        has_params = True
    else:
        optimizer = None
        scheduler = None
        has_params = False
    
    history = {
        'train_loss': [], 'train_p': [], 'train_ar': [], 'train_energy': [], 'train_smooth': [], 'train_order': [], 'train_p_acc': [],
        'val_loss': [], 'val_p': [], 'val_ar': [], 'val_energy': [], 'val_smooth': [], 'val_order': [], 'val_p_acc': []
    }
    
    pbar = tqdm(total=n_epochs, desc='Training')
    for epoch in range(n_epochs):
        # Train
        model.train()
        train_losses = {'total': [], 'p': [], 'ar': [], 'energy': [], 'smooth': [], 'order': []}
        train_correct = 0
        train_total = 0
        
        for x_batch, p_batch in train_loader:
            x_batch = x_batch.to(device)
            p_batch = p_batch.to(device)
            
            if has_params:
                optimizer.zero_grad()
            
            coeffs, p_logits, p_hard, x_hat = model(x_batch)
            
            l_p = loss_p(p_logits, p_batch)
            l_ar = loss_ar(x_batch, x_hat, p_max)
            l_energy = loss_energy(x_hat, P0, W)
            l_smooth = loss_smooth(coeffs)
            l_order = loss_order(p_logits)
            
            total_loss = lambda_p * l_p + lambda_ar * l_ar + lambda_energy * l_energy + lambda_smooth * l_smooth + lambda_order * l_order
            
            if has_params:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            train_losses['total'].append(total_loss.item())
            train_losses['p'].append(l_p.item())
            train_losses['ar'].append(l_ar.item())
            train_losses['energy'].append(l_energy.item())
            train_losses['smooth'].append(l_smooth.item())
            train_losses['order'].append(l_order.item())
            
            train_correct += (p_hard == p_batch).sum().item()
            train_total += p_batch.shape[0]
        
        # Validate
        model.eval()
        val_losses = {'total': [], 'p': [], 'ar': [], 'energy': [], 'smooth': [], 'order': []}
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for x_batch, p_batch in val_loader:
                x_batch = x_batch.to(device)
                p_batch = p_batch.to(device)
                
                coeffs, p_logits, p_hard, x_hat = model(x_batch)
                
                l_p = loss_p(p_logits, p_batch)
                l_ar = loss_ar(x_batch, x_hat, p_max)
                l_energy = loss_energy(x_hat, P0, W)
                l_smooth = loss_smooth(coeffs)
                l_order = loss_order(p_logits)
                
                total_loss = lambda_p * l_p + lambda_ar * l_ar + lambda_energy * l_energy + lambda_smooth * l_smooth + lambda_order * l_order
                
                val_losses['total'].append(total_loss.item())
                val_losses['p'].append(l_p.item())
                val_losses['ar'].append(l_ar.item())
                val_losses['energy'].append(l_energy.item())
                val_losses['smooth'].append(l_smooth.item())
                val_losses['order'].append(l_order.item())
                
                val_correct += (p_hard == p_batch).sum().item()
                val_total += p_batch.shape[0]
        
        history['train_loss'].append(np.mean(train_losses['total']))
        history['train_p'].append(np.mean(train_losses['p']))
        history['train_ar'].append(np.mean(train_losses['ar']))
        history['train_energy'].append(np.mean(train_losses['energy']))
        history['train_smooth'].append(np.mean(train_losses['smooth']))
        history['train_order'].append(np.mean(train_losses['order']))
        history['train_p_acc'].append(train_correct / train_total)
        
        history['val_loss'].append(np.mean(val_losses['total']))
        history['val_p'].append(np.mean(val_losses['p']))
        history['val_ar'].append(np.mean(val_losses['ar']))
        history['val_energy'].append(np.mean(val_losses['energy']))
        history['val_smooth'].append(np.mean(val_losses['smooth']))
        history['val_order'].append(np.mean(val_losses['order']))
        history['val_p_acc'].append(val_correct / val_total)
        
        if has_params:
            scheduler.step(history['val_loss'][-1])
        
        
        pbar.update(1)
        pbar.set_postfix(
            train=f"{history['train_loss'][-1]:.4f}",
            val=f"{history['val_loss'][-1]:.4f}",
            p_acc=f"{history['val_p_acc'][-1]:.3f}"
        )
    
    pbar.close()
    return history