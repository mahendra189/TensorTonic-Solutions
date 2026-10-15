
import torch

def linear_attention_duality(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor
) -> dict:
    # Use float32 for intermediate calculations
    qf = q.float()
    kf = k.float()
    vf = v.float()

    B, S, Dk = k.shape
    Dv = v.shape[-1]

    # Each write: k_t @ v_t.T
    # Shape: (B, S, Dk, Dv)
    writes = kf.unsqueeze(-1) * vf.unsqueeze(-2)

    # Parallel formulation: inclusive prefix sum of state writes
    states = torch.cumsum(writes, dim=1)

    # Read each state using its corresponding query
    # q_t^T @ S_t -> (B, S, Dv)
    parallel_output = torch.einsum("bsd,bsdv->bsv", qf, states)

    # Recurrent formulation: update state one token at a time
    state = torch.zeros(
        (B, Dk, Dv),
        dtype=torch.float32,
        device=q.device
    )

    outputs = []

    for t in range(S):
        kt = kf[:, t, :]
        vt = vf[:, t, :]
        qt = qf[:, t, :]

        # S_t = S_(t-1) + k_t v_t^T
        state = state + kt.unsqueeze(-1) * vt.unsqueeze(-2)

        # y_t = q_t^T S_t
        yt = torch.einsum("bd,bdv->bv", qt, state)
        outputs.append(yt)

    recurrent_output = torch.stack(outputs, dim=1)

    return {
        "parallel_output": parallel_output.to(q.dtype),
        "recurrent_output": recurrent_output.to(q.dtype),
        "final_state": state.to(q.dtype)
    }
