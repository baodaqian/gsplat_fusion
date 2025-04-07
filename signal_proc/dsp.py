import torch

def get_array_pos(theta, phi, array_dist, spacing, num_rx, num_tx, device):
    """
    Compute the absolute positions of the RX and TX antennas.
    
    Args:
        theta (torch.Tensor): Elevation angle(s) tensor.
        phi (torch.Tensor): Azimuth angle(s) tensor.
        array_dist (float): Array distance.
        spacing (float): Antenna spacing.
        num_rx (int): Number of RX antennas.
        num_tx (int): Number of TX antennas.
        device (str): Device to use.
    
    Returns:
        tuple: (rx_positions, tx_positions)
    """
    theta = theta.squeeze(0)[0]
    phi = phi.squeeze(0)[0]
    r_0 = array_dist * torch.tensor([
        torch.sin(theta) * torch.cos(phi),
        torch.sin(theta) * torch.sin(phi),
        torch.cos(theta)
    ], device=device)
    rx_positions = torch.zeros(num_rx, 3, device=device)
    tx_positions = torch.zeros(num_tx, 3, device=device)
    for n in range(num_rx):
        offset = spacing * (n + 0.5) * torch.tensor([-torch.sin(phi), torch.cos(phi), 0], device=device)
        rx_positions[n] = r_0 + offset
    for n in range(num_tx):
        offset = spacing * (n + 0.5) * torch.tensor([
            -torch.cos(theta) * torch.cos(phi),
            -torch.cos(theta) * torch.sin(phi),
             torch.sin(theta)
        ], device=device)
        tx_positions[n] = r_0 + offset
    return rx_positions, tx_positions

def forward_operator(selected_freqs, k_vector, rx_pos, tx_pos, scatterer_pos, scatterer_weights):
    """
    Parallel forward operator: projects scatterer weights to S-parameters.
    
    Args:
        selected_freqs (torch.Tensor): Selected frequency samples.
        k_vector (torch.Tensor): k-vector computed from frequencies.
        rx_pos (torch.Tensor): Receiver positions.
        tx_pos (torch.Tensor): Transmitter positions.
        scatterer_pos (torch.Tensor): Flattened grid of scatterer positions.
        scatterer_weights (torch.Tensor): Complex scatterer weights.
    
    Returns:
        torch.Tensor: S-parameters (complex tensor of shape [num_rx, num_tx, nf]).
    """
    device = selected_freqs.device
    num_rx = rx_pos.shape[0]
    num_tx = tx_pos.shape[0]
    nf = selected_freqs.shape[0]
    
    S_pars = torch.zeros((num_rx, num_tx, nf), dtype=torch.cfloat, device=device)
    
    # Compute distances from scatterers to antennas (broadcasting over scatterers)
    scat_dist_rx = scatterer_pos.unsqueeze(1) - rx_pos.unsqueeze(0)  # [N_scatterers, num_rx, 3]
    scat_dist_tx = scatterer_pos.unsqueeze(1) - tx_pos.unsqueeze(0)  # [N_scatterers, num_tx, 3]
    
    # Expand dimensions for summing contributions over scatterers
    scat_dist_rx = scat_dist_rx.unsqueeze(2)  # [N_scatterers, num_rx, 1, 3]
    scat_dist_tx = scat_dist_tx.unsqueeze(1)  # [N_scatterers, 1, num_tx, 3]
    
    # Compute two-way path lengths
    scat_dist_2way = torch.norm(scat_dist_rx + scat_dist_tx, dim=-1)  # [N_scatterers, num_rx, num_tx]
    
    scat_mag = torch.abs(scatterer_weights).view(-1, 1, 1)
    scat_angle = torch.angle(scatterer_weights).view(-1, 1, 1)
    
    for f in range(nf):
        phase_diff = scat_dist_2way * k_vector[f]
        phase_diff += scat_angle  # Incorporate scatterer phase
        S_contrib = torch.sum(scat_mag * torch.exp(1j * phase_diff), dim=0)
        S_pars[:, :, f] = S_contrib
    return S_pars

def backward_operator(S_pars, grid, rx_positions, tx_positions, freqs):
    """
    Parallel backward operator: reconstructs an image (or volume) from S-parameters.
    
    Args:
        S_pars (torch.Tensor): S-parameters (complex tensor of shape [num_rx, num_tx, nf]).
        grid (torch.Tensor): Grid points (flattened or arranged).
        rx_positions (torch.Tensor): RX antenna positions.
        tx_positions (torch.Tensor): TX antenna positions.
        freqs (torch.Tensor): Frequency tensor.
    
    Returns:
        torch.Tensor: Reconstructed image (complex tensor).
    """
    device = S_pars.device
    grid = grid.to(device)
    rx_positions = rx_positions.to(device)
    tx_positions = tx_positions.to(device)
    freqs = freqs.to(device)
    nfreqs = S_pars.shape[2]
    im = torch.zeros(grid.shape[0], dtype=torch.cfloat, device=device)
    
    # Compute distances from each grid point to each antenna
    rx_dists = torch.norm(grid.unsqueeze(1) - rx_positions.unsqueeze(0), dim=-1)
    tx_dists = torch.norm(grid.unsqueeze(1) - tx_positions.unsqueeze(0), dim=-1)
    dists = rx_dists.unsqueeze(2) + tx_dists.unsqueeze(1)
    
    for f in range(nfreqs):
        k_val = get_kvector(freqs)[f]
        U_arg = k_val * dists
        U = torch.cos(U_arg) - 1j * torch.sin(U_arg)
        S = S_pars[:, :, f]
        im += (U * S.unsqueeze(0)).sum((1, 2))
    return im

def forward_lessparallel(freqs, kvector, arr_pos_rx, arr_pos_tx, scatterer_pos, scatterer_weights):
    """
    Less-parallel forward operator for reduced RAM consumption.
    
    Args:
        freqs (torch.Tensor): Frequency tensor.
        kvector (torch.Tensor): k-vector computed from freqs.
        arr_pos_rx (torch.Tensor): Receiver positions.
        arr_pos_tx (torch.Tensor): Transmitter positions.
        scatterer_pos (torch.Tensor): Scatterer positions.
        scatterer_weights (torch.Tensor): Complex scatterer weights.
    
    Returns:
        torch.Tensor: S-parameters (complex tensor of shape [num_rx, num_tx, nf]).
    """
    device = freqs.device
    num_rx = arr_pos_rx.shape[0]
    num_tx = arr_pos_tx.shape[0]
    nf = freqs.shape[0]
    
    S_pars = torch.zeros((num_rx, num_tx, nf), dtype=torch.cfloat, device=device)
    
    # Compute two-way distances in a less-parallelized loop over frequencies
    scat_dist_rx = scatterer_pos.unsqueeze(1) - arr_pos_rx.unsqueeze(0)
    scat_dist_tx = scatterer_pos.unsqueeze(1) - arr_pos_tx.unsqueeze(0)
    scat_dist_rx = scat_dist_rx.unsqueeze(2)
    scat_dist_tx = scat_dist_tx.unsqueeze(1)
    scat_dist_2way = torch.norm(scat_dist_rx + scat_dist_tx, dim=-1)
    
    scat_mag = torch.abs(scatterer_weights).view(-1, 1, 1)
    scat_angle = torch.angle(scatterer_weights).view(-1, 1, 1)
    
    for f in range(nf):
        phase_diff = scat_dist_2way * kvector[f]
        phase_diff += scat_angle
        S_contrib = torch.sum(scat_mag * torch.exp(1j * phase_diff), dim=0)
        S_pars[:, :, f] = S_contrib
    return S_pars

def backward_lessparallel(S_pars, grid, rx_positions, tx_positions, freqs):
    """
    Less-parallel backward operator for reduced RAM consumption.
    
    Args:
        S_pars (torch.Tensor): S-parameters (complex tensor of shape [num_rx, num_tx, nf]).
        grid (torch.Tensor): Grid points.
        rx_positions (torch.Tensor): Receiver positions.
        tx_positions (torch.Tensor): Transmitter positions.
        freqs (torch.Tensor): Frequency tensor.
    
    Returns:
        torch.Tensor: Reconstructed image (complex tensor).
    """
    device = S_pars.device
    grid = grid.to(device)
    rx_positions = rx_positions.to(device)
    tx_positions = tx_positions.to(device)
    freqs = freqs.to(device)
    nfreqs = S_pars.shape[2]
    im = torch.zeros(grid.shape[0], dtype=torch.cfloat, device=device)
    
    rx_dists = torch.norm(grid.unsqueeze(1) - rx_positions.unsqueeze(0), dim=-1)
    tx_dists = torch.norm(grid.unsqueeze(1) - tx_positions.unsqueeze(0), dim=-1)
    dists = rx_dists.unsqueeze(2) + tx_dists.unsqueeze(1)
    
    for f in range(nfreqs):
        k_val = get_kvector(freqs)[f]
        U_arg = k_val * dists
        U = torch.cos(U_arg) - 1j * torch.sin(U_arg)
        S = S_pars[:, :, f]
        im += (U * S.unsqueeze(0)).sum((1, 2))
    return im