import torch

# coeffs for polar express 
# not pre_computed, same as modded-nanoGPT 
coeffs_list = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323)
]

@torch.compile()
def zeropower_polar_express(G:torch.Tensor, steps: int = 5):
    """Polar express as replacement for Newton-Schulz iteration"""
    assert G.ndim >= 2
    assert steps <= len(coeffs_list)

    X = G.bfloat16()

    transpose_needed = G.size(-2) > G.size(-1) # transposing if tall matrix
    if transpose_needed: 
        X = X.mT 
    
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7) # safety factor
    
    coeffs = coeffs_list[:steps]
    for a, b, c in coeffs:
        A = X @ X.mT 
        A2 = A @ A 
        B = b * A + c * A2
        X = a * X + B @ X  # Right-multiplication for left polar factor
    
    if transpose_needed: 
        X = X.mT 
    
    return X # orthogonalized 


class Muon(torch.optim.Optimizer):
    """Muon - MomentUm Orthogonalized by Polar Express / Newton Schulz"""
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                g = zeropower_polar_express(g, steps=group["ns_steps"]) # steps are 5 for both ns and pe
                g = g.to(p.dtype)
                p.add_(g.view_as(p), alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)


class Teon(torch.optim.Optimizer):
    """Teon - Tensorized Orthonormalization (generalization of Muon)"""
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, k=2):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, k=k)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            k = group['k']
            params = group["params"]

            # Process in chunks of k
            for i in range(0, len(params), k):
                p_list = params[i:i+k]
                
                # If remaining params are not enough to form a group of k, fallback to regular MUON
                if len(p_list) != k:
                    for p in p_list:
                        if p.grad is None:
                            continue
                        g = p.grad
                        state = self.state[p]

                        if "momentum_buffer" not in state:
                            state["momentum_buffer"] = torch.zeros_like(g)

                        buf = state["momentum_buffer"]
                        buf.lerp_(g, 1 - group["momentum"])
                        g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                        g = zeropower_polar_express(g, steps=group["ns_steps"]) # steps are 5 for both ns and pe
                        g = g.to(p.dtype)
                        p.add_(g.view_as(p), alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)
                    continue

                grads = []
                mems = []
                valid = True
                for p in p_list:
                    if p.grad is None:
                        valid = False
                        break
                    grads.append(p.grad)
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p.grad)
                    mems.append(state["momentum_buffer"])
                
                if not valid:
                    continue
                
                for mem, grad in zip(mems, grads):
                    mem.lerp_(grad, 1 - group["momentum"])
                
                if group["nesterov"]:
                    gs = [g.lerp_(m, group["momentum"]) for g, m in zip(grads, mems)]
                else:
                    gs = [m for m in mems]
                
                # TEON works best when stacking similar matrices across layers.
                # If weights are combined QKV (3*D x D), we split them to avoid mixing head correlations.
                first_g = gs[0]
                if first_g.size(0) == first_g.size(1) * 3:
                    d = first_g.size(1)
                    q_list, k_list, v_list = [], [], []
                    for g in gs:
                        q, k_v, v = g.split(d, dim=0)
                        q_list.append(q)
                        k_list.append(k_v)
                        v_list.append(v)
                    
                    # Orthogonalize Q, K, and V tensors separately
                    # Each has shape (D, D, k) -> Matricized to (D, D*k)
                    
                    # Process Query matrices
                    T_q = torch.stack(q_list, dim=-1)
                    T_q_m1 = T_q.reshape(d, d * k)
                    O_q_m1 = zeropower_polar_express(T_q_m1, steps=group["ns_steps"])
                    O_q = O_q_m1.reshape(d, d, k)
                    
                    # Process Key matrices
                    T_k = torch.stack(k_list, dim=-1)
                    T_k_m1 = T_k.reshape(d, d * k)
                    O_k_m1 = zeropower_polar_express(T_k_m1, steps=group["ns_steps"])
                    O_k = O_k_m1.reshape(d, d, k)
                    
                    # Process Value matrices
                    T_v = torch.stack(v_list, dim=-1)
                    T_v_m1 = T_v.reshape(d, d * k)
                    O_v_m1 = zeropower_polar_express(T_v_m1, steps=group["ns_steps"])
                    O_v = O_v_m1.reshape(d, d, k)
                    
                    # Recombine and set scale (for square matrices scale is 1.0)
                    O = torch.cat([O_q, O_k, O_v], dim=0)
                    scale = 1.0
                else:
                    # General case: stack whole matrices as (m, n, k) tensor
                    T = torch.stack(gs, dim=-1)
                    m, n, _ = T.shape
                    # Mode-1 matricization: m x (n * k)
                    T_m1 = T.reshape(m, n * k)
                    O_m1 = zeropower_polar_express(T_m1, steps=group["ns_steps"])
                    O = O_m1.reshape(m, n, k)
                    scale = max(1, m / n)**0.5
                
                for idx, p in enumerate(p_list):
                    update = O[:, :, idx].to(p.device).to(p.dtype)
                    p.add_(update.view_as(p), alpha=-group["lr"] * scale)


def get_optimizer_groups(model, use_teon=False, teon_k=2):
    """
    Group params into TEON, MUON, and AdamW groups.
    Based on TEON paper ablations, TEON works best on QKV parameters (in_proj_weight) of consecutive layers.
    MUON works best on other 2D matrices (out_proj, mlp).
    AdamW is used for everything else (biases, norms, embeddings).
    """
    adam_params = []
    muon_params = []
    teon_params = []
    
    in_proj_weights = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        
        # ViT QKV projection weights
        if "in_proj_weight" in name:
            in_proj_weights.append((name, p))
        elif p.ndim == 2:
            muon_params.append(p)
        else:
            adam_params.append(p)
            
    if use_teon:
        import re
        def get_layer_idx(n):
            match = re.search(r'encoder_layer_(\d+)', n)
            if match:
                return int(match.group(1))
            return 0
        
        # Sort in_proj_weights by layer index to ensure consecutive layers are grouped
        in_proj_weights = sorted(in_proj_weights, key=lambda x: get_layer_idx(x[0]))
        
        for name, p in in_proj_weights:
            teon_params.append(p)
    else:
        for name, p in in_proj_weights:
            muon_params.append(p)
            
    return teon_params, muon_params, adam_params
