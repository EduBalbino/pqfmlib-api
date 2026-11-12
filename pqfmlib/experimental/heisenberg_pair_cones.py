"""Experimental backward-cone simulation for local and diagonal pair observables."""
import numpy as np
import torch


def pair_cones(gates, num_qubits, edges):
    """Return exact backward cones and local coordinates for every requested target."""
    targets = [(site,) for site in range(num_qubits)] + list(edges)
    cones, lookup, cone_ids, local_targets = [], {}, [], []
    for target in targets:
        active = set(target)
        selected = []
        for index in range(len(gates) - 1, -1, -1):
            sites = gates[index][0]
            if active.intersection(sites):
                active.update(sites)
                selected.append(index)
        selected.reverse()
        positions = {site: i for i, site in enumerate(sorted(active))}
        key = (tuple(positions), tuple(selected))
        if key not in lookup:
            lookup[key] = len(cones)
            cones.append((positions, selected))
        cone_ids.append(lookup[key])
        local_targets.append(tuple(positions[site] for site in target))
    return cones, cone_ids, local_targets, max(2, max(len(c[0]) for c in cones))


def simulate_pair_cones(gates, num_qubits, batch, edges, plan, device="cuda"):
    """Evolve independent cones; output Z/X/Y per site followed by ZZ/XX/YY per edge."""
    from pqfmlib.experimental.heisenberg_tilelang import apply_cone_gate

    cones, cone_ids, targets, width = plan
    size = 1 << width
    depth = max(len(selected) for _, selected in cones)
    identity = np.broadcast_to(np.eye(4, dtype=complex), (batch, 4, 4))
    matrices, matrix_ids, sites_by_step = [identity], {}, []
    for step in range(depth):
        cone_sites = []
        for positions, selected in cones:
            if step >= len(selected):
                local_sites, gate_id = (0, 1), 0
            else:
                sites, _, matrix = gates[selected[step]]
                local_sites = tuple(positions[site] for site in sites)
                key = (id(matrix), len(sites))
                if key not in matrix_ids:
                    matrix_ids[key] = len(matrices)
                    matrices.append(np.kron(np.eye(2), matrix) if len(sites) == 1 else matrix)
                gate_id = matrix_ids[key]
                if len(sites) == 1:
                    local_sites = (local_sites[0], 0 if local_sites[0] else 1)
            cone_sites.append((*local_sites, gate_id))
        sites_by_step.append(cone_sites)
    parity = all(len(gates[index][0]) == 2 and gates[index][1]
                 for _, selected in cones for index in selected)
    matrix_tensor = torch.view_as_real(torch.tensor(np.stack(matrices), device=device)).contiguous()
    site_tensor = torch.tensor(sites_by_step, dtype=torch.int32, device=device)
    state = torch.zeros((batch * len(cones), size, 2), dtype=torch.float64, device=device)
    state[:, 0, 0] = 1.0
    for step in range(depth):
        output = torch.empty_like(state)
        apply_cone_gate(size, batch, len(cones), len(matrices), parity)(state, matrix_tensor, site_tensor[step], output)
        state = output
    states = state.reshape(len(cones), batch, size, 2)
    device = state.device
    local_states = states.index_select(0, torch.tensor(cone_ids[:num_qubits], device=device))
    local_bits = torch.tensor([t[0] for t in targets[:num_qubits]], device=device)[:, None]
    half = torch.arange(size // 2, device=device)[None, :]
    zero = (half & ((1 << local_bits) - 1)) | ((half >> local_bits) << (local_bits + 1))
    one = zero | (1 << local_bits)
    shape = (num_qubits, batch, size // 2, 2)
    a = local_states.gather(2, zero[:, None, :, None].expand(shape))
    b = local_states.gather(2, one[:, None, :, None].expand(shape))
    ar, ai, br, bi = a[..., 0], a[..., 1], b[..., 0], b[..., 1]
    local = torch.stack(((ar*ar+ai*ai-br*br-bi*bi).sum(dim=2),
                         (2*(ar*br+ai*bi)).sum(dim=2),
                         (2*(ar*bi-ai*br)).sum(dim=2)), dim=2)
    pair_states = states.index_select(0, torch.tensor(cone_ids[num_qubits:], device=device))
    pair_bits = torch.tensor(targets[num_qubits:], device=device)
    left, right = pair_bits[:, :1], pair_bits[:, 1:]
    basis = torch.arange(size, device=device)[None, :]
    sign = (1-2*((basis >> left)&1)) * (1-2*((basis >> right)&1))
    flipped_indices = basis ^ (1 << left) ^ (1 << right)
    flipped = pair_states.gather(2, flipped_indices[:, None, :, None].expand(len(edges), batch, size, 2))
    overlap = pair_states[..., 0]*flipped[..., 0] + pair_states[..., 1]*flipped[..., 1]
    probabilities = pair_states[..., 0]**2 + pair_states[..., 1]**2
    pairs = torch.stack(((probabilities*sign[:, None, :]).sum(dim=2), overlap.sum(dim=2),
                         -(overlap*sign[:, None, :]).sum(dim=2)), dim=2)
    return torch.cat((local.permute(1, 0, 2).reshape(batch, -1),
                      pairs.permute(1, 0, 2).reshape(batch, -1)), dim=1).cpu().numpy()
