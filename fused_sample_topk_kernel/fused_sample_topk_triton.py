import torch
import triton
import triton.language as tl


@triton.jit
def fused_sample_topk_kernel(
    logits_ptr,              # [B, V]
    output_ptr,              # [B]
    temp,
    top_k: tl.constexpr,
    B: tl.constexpr,
    V: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_id = tl.program_id(0)
    offset = batch_id * V

    logits = tl.load(logits_ptr + offset + tl.arange(0, BLOCK_SIZE))

    logits /= temp

    lmax = tl.max(logits, axis=0)
    logits -= lmax

    probs = tl.exp(logits)
    total = tl.sum(probs, axis=0)
    probs /= total

    sorted_probs, sorted_idx = tl.sort(probs, axis=0, descending=True)

    topk_probs = sorted_probs[:top_k]
    topk_idx = sorted_idx[:top_k]

    topk_probs = topk_probs / tl.sum(topk_probs, axis=0)

    u = tl.rand([1], seed=batch_id)
    cdf = tl.cumsum(topk_probs, axis=0)
    sampled_idx = tl.sum(cdf < u, axis=0)

    token = topk_idx[sampled_idx]

    tl.store(output_ptr + batch_id, token)


def fused_sample_topk(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 10):
    """
    logits: [B, V] float32 or float16
    returns: [B] int32 tokens
    """
    B, V = logits.shape
    logits = logits.contiguous()

    output = torch.empty(B, dtype=torch.int32, device=logits.device)

    # TODO improve for larger BLOCK_SIZE
    BLOCK_SIZE = V  

    fused_sample_topk_kernel[
        B
    ](
        logits, output,
        temperature,
        top_k,
        B, V,
        BLOCK_SIZE
    )

    return output
