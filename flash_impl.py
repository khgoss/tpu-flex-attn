import math
import jax
from functools import partial
from jax import nn
from jax import custom_vjp
from jax import numpy as jnp, lax, jit
from jax.numpy import einsum

from einops import rearrange

import jax.profiler
LOG_DIR = "/tmp/impl_prof"

MASK_VALUE = -1e10

VMEM_SZ = 2**10 # want each tile to fit in vmem
Q_TILE_SZ = 2**10
K_TILE_SZ = 2**10

'''
TODOs
-   add default param for mask n skip masking logic if None (would this be better? idk)
-   try impl that does recompute O for bkwd pass
'''


# -------------- FORWARD PASS ----------------------------------
def tile_scans(q, k, v, mask):
    """
    NB: Inputs may be padded!
    q: (q_len, batch, heads, head_dim)
    k: (k_len, batch, heads, head_dim)
    v: (k_len, batch, heads, v_dim)
    mask: (k_len, batch)
    """
    q_len, batch, heads, head_dim = q.shape
    k_len, v_dim = k.shape[0], v.shape[-1]

    q_scaled = q * (1/jnp.sqrt(head_dim))

    def tile_scan(carries, tile_idx):
        result, row_sum, row_max = carries
        
        k_tile = lax.dynamic_slice(k, (tile_idx, 0, 0, 0),
                    slice_sizes=(K_TILE_SZ, batch, heads, head_dim))
        v_tile = lax.dynamic_slice(v, (tile_idx, 0, 0, 0),
            slice_sizes=(K_TILE_SZ, batch, heads, v_dim))
        mask_tile = lax.dynamic_slice(mask, (tile_idx, 0),
            slice_sizes=(K_TILE_SZ, batch))

        # compute weights - Q_tile*K_tile^T
        weights = einsum('i b h d, j b h d -> i b h j', q_scaled, k_tile)

        # apply mask
        mask_tile = rearrange(mask_tile, 'j b -> 1 b 1 j')
        weights = jnp.where(mask_tile, weights, MASK_VALUE)

        # find max in tile, new global row max
        tile_row_max = jnp.max(weights, axis=-1, keepdims=True)
        row_max_update = jnp.maximum(tile_row_max, row_max)

        # sum e^(x_i)*v_i / sum e(x_i)
        exp_weights = jnp.exp(weights - row_max_update)
        exp_weights = jnp.where(mask_tile, exp_weights, 0.)

        # update row sum
        tile_row_sum = jnp.sum(exp_weights, axis=-1, keepdims=True)
        exp_row_max_delta = jnp.exp(row_max - row_max_update)
        row_sum_update = exp_row_max_delta * row_sum + tile_row_sum

        # weighted sum of values
        exp_vals = einsum('i ... j, j ... d -> i ... d', exp_weights, v_tile)
        
        # rescale old numerator, add new update, normalize again
        result = exp_row_max_delta*row_sum/row_sum_update*result + exp_vals*(1.0/row_sum_update)

        return (result, row_sum_update, row_max_update), None

    result = jnp.zeros((q_len, batch, heads, v_dim))
    row_sum = jnp.zeros((q_len, batch, heads, 1))
    row_max = jnp.ones((q_len, batch, heads, 1))*-1e6

    (result, row_sum, row_max), _ = lax.scan(
        tile_scan, 
        init=(result, row_sum, row_max), 
        xs=jnp.arange(0, k_len, K_TILE_SZ),
        length=math.ceil(k_len/K_TILE_SZ)
    )

    row_sum = rearrange(row_sum, 'n ... 1 -> n ...')
    row_max = rearrange(row_max, 'n ... 1 -> n ...')

    # need LSE for backward pass
    lse = jnp.log(row_sum) + row_max
    return result, lse


def flash_attn(q, k, v, mask):
    """
    q: (batch, heads, q_len, dim)
    k: (batch, heads, k_len, dim)
    v: (batch, heads, k_len, v_dim)
    mask: (batch, k_len)
    """
    batch, heads, q_len, head_dim = q.shape
    k_len = k.shape[2]
    v_dim = v.shape[-1]

    # seq first layout
    q, k, v = map(lambda t: rearrange(t, 'b h n d -> n b h d'), (q, k, v))
    mask = rearrange(mask, 'b j -> j b')

    # pad q, k, v and mask if needed
    # this is needed bc dynamic_slice needs static tile sizes
    q_len_padded = math.ceil(q_len / Q_TILE_SZ) * Q_TILE_SZ
    if q_len_padded > q_len:
        padding = q_len_padded - q_len
        q = jnp.pad(q, ((0, padding), (0, 0), (0, 0), (0, 0)), mode='constant')

    k_len_padded = math.ceil(k_len / K_TILE_SZ) * K_TILE_SZ
    if k_len_padded > k_len:
        padding = k_len_padded - k_len
        k = jnp.pad(k, ((0, padding), (0, 0), (0, 0), (0, 0)), mode='constant')
        v = jnp.pad(v, ((0, padding), (0, 0), (0, 0), (0, 0)), mode='constant')
        # mask padded positions (ie, don't attend)
        mask = jnp.pad(mask, ((0, padding), (0, 0)), mode='constant', constant_values=False)

    # scan over q tiles, apply k tiles to each
    def tile_scanner(carries, tile_idx):
        q_tile = lax.dynamic_slice(q, (tile_idx, 0, 0, 0), 
                                   slice_sizes=(Q_TILE_SZ, batch, heads, head_dim))
        out_tile, lse_tile = tile_scans(q_tile, k, v, mask)
        return None, (out_tile, lse_tile)

    _, (out, lse) = lax.scan(
        tile_scanner, 
        init=None, 
        xs=jnp.arange(0, q_len_padded, Q_TILE_SZ), # nb: xs handles tile indices
        length=math.ceil(q_len_padded/Q_TILE_SZ)
    )
    
    # concatenate, remove padding
    out = rearrange(out, 'c n b h d -> b h (c n) d')
    lse = rearrange(lse, 'c n b h -> b h (c n)')
    
    # get back to og length
    out = out[:, :, :q_len, :]
    lse = lse[:, :, :q_len]

    return out, lse


# -------------- BACKWARD PASS -------------------------------
# nb -- recomputing is faster than just storing the old vals
def compute_p(q, k_tile, lse_tile, mask_tile):
    scores = einsum('i b h d, j b h d -> i b h j', q, k_tile)
    scores = jnp.where(mask_tile, scores, MASK_VALUE)
    p = jnp.exp(scores - lse_tile)
    p = jnp.where(mask_tile, p, 0.)
    return p 

@jit
def flash_attention_bckwd(res, dout):
    """
    q: (batch, heads, q_len, dim)
    k: (batch, heads, k_len, dim)
    v: (batch, heads, k_len, v_dim)
    mask: (batch, k_len)
    """
    q, k, v, mask, o, lse = res
    batch, heads, q_len, head_dim = q.shape
    k_len = k.shape[2]

    # seq first
    q, k, v, o, dout = map(lambda t: rearrange(t, 'b h n d -> n b h d'), (q, k, v, o, dout))
    mask = rearrange(mask, 'b j -> j b')
    lse = rearrange(lse, 'b h n -> n b h 1')

    scale = 1 / jnp.sqrt(head_dim)

    # pad again...
    q_len_padded = math.ceil(q_len / Q_TILE_SZ) * Q_TILE_SZ
    k_len_padded = math.ceil(k_len / K_TILE_SZ) * K_TILE_SZ
    
    if q_len_padded > q_len:
        pad_q = q_len_padded - q_len
        q = jnp.pad(q, ((0, pad_q), (0, 0), (0, 0), (0, 0)))
        o = jnp.pad(o, ((0, pad_q), (0, 0), (0, 0), (0, 0)))
        dout = jnp.pad(dout, ((0, pad_q), (0, 0), (0, 0), (0, 0)))
        lse = jnp.pad(lse, ((0, pad_q), (0, 0), (0, 0), (0, 0)))
    
    if k_len_padded > k_len:
        pad_k = k_len_padded - k_len
        k = jnp.pad(k, ((0, pad_k), (0, 0), (0, 0), (0, 0)))
        v = jnp.pad(v, ((0, pad_k), (0, 0), (0, 0), (0, 0)))
        mask = jnp.pad(mask, ((0, pad_k), (0, 0)), constant_values=False)

    dq = jnp.zeros_like(q)
    dk = jnp.zeros_like(k)
    dv = jnp.zeros_like(v)

    def q_tile_scan(carry, q_tile_idx):
        dq_acc, dk_acc, dv_acc = carry

        q_tile = lax.dynamic_slice(q, (q_tile_idx, 0, 0, 0), 
                                   (Q_TILE_SZ, batch, heads, head_dim))
        o_tile = lax.dynamic_slice(o, (q_tile_idx, 0, 0, 0), 
                                   (Q_TILE_SZ, batch, heads, head_dim))
        dout_tile = lax.dynamic_slice(dout, (q_tile_idx, 0, 0, 0), 
                                      (Q_TILE_SZ, batch, heads, head_dim))
        lse_tile = lax.dynamic_slice(lse, (q_tile_idx, 0, 0, 0), 
                                     (Q_TILE_SZ, batch, heads, 1))

        # compute D = dO dot O (delta correction term)
        # nb: using saved o to do so rather than recomputing
        D = jnp.sum(dout_tile * o_tile, axis=-1, keepdims=True)

        # scale q
        q_scaled = q_tile * scale

        def kv_tile_scan(kv_carry, kv_tile_idx):
            dq_tile_acc, dk_tile, dv_tile = kv_carry

            k_tile = lax.dynamic_slice(k, (kv_tile_idx, 0, 0, 0), 
                                       (K_TILE_SZ, batch, heads, head_dim))
            v_tile = lax.dynamic_slice(v, (kv_tile_idx, 0, 0, 0), 
                                       (K_TILE_SZ, batch, heads, head_dim))
            mask_tile = lax.dynamic_slice(mask, (kv_tile_idx, 0), 
                                         (K_TILE_SZ, batch))
            mask_tile = rearrange(mask_tile, 'j b -> 1 b 1 j')

            # recomput p for this tile
            p = compute_p(q_scaled, k_tile, lse_tile, mask_tile)

            # compute dV contribution for this k/v tile
            dv_chunk = einsum('i b h j, i b h d -> j b h d', p, dout_tile)
            dv_tile = lax.dynamic_update_slice(dv_tile, dv_chunk, (kv_tile_idx, 0, 0, 0))

            # compute dP
            dp = einsum('i b h d, j b h d -> i b h j', dout_tile, v_tile)

            # gradient wrt attention logits (pre-softmax)
            ds = p * scale * (dp - D)

            # get results from all kv tiles
            dq_tile_acc += einsum('i b h j, j b h d -> i b h d', ds, k_tile)

            # compute dK contribution for this kv tile
            dk_chunk = einsum('i b h j, i b h d -> j b h d', ds, q_tile)
            dk_tile = lax.dynamic_update_slice(dk_tile, dk_chunk, (kv_tile_idx, 0, 0, 0))

            return (dq_tile_acc, dk_tile, dv_tile), None

        # init for this q tile
        dq_tile_init = jnp.zeros_like(q_tile)
        dk_tile_init = jnp.zeros_like(k)
        dv_tile_init = jnp.zeros_like(v)

        # scan over kv tiles !!
        (dq_tile_final, dk_update, dv_update), _ = lax.scan(
            kv_tile_scan,
            (dq_tile_init, dk_tile_init, dv_tile_init),
            jnp.arange(0, k_len_padded, K_TILE_SZ),
            length=math.ceil(k_len_padded / K_TILE_SZ)
        )

        # nb: each q tile's gradient is independent, can write directly
        dq_acc = lax.dynamic_update_slice(dq_acc, dq_tile_final, (q_tile_idx, 0, 0, 0))
        dk_acc += dk_update
        dv_acc += dv_update

        return (dq_acc, dk_acc, dv_acc), None

    # scan over all q tiles
    (dq, dk, dv), _ = lax.scan(
        q_tile_scan,
        (dq, dk, dv),
        jnp.arange(0, q_len_padded, Q_TILE_SZ),
        length=math.ceil(q_len_padded / Q_TILE_SZ)
    )

    # remove padding, convert back to batch first format from seq first
    dq = dq[:q_len, :, :, :]
    dk = dk[:k_len, :, :, :]
    dv = dv[:k_len, :, :, :]
    
    dq = rearrange(dq, 'n b h d -> b h n d')
    dk = rearrange(dk, 'n b h d -> b h n d')
    dv = rearrange(dv, 'n b h d -> b h n d')

    return dq, dk, dv, None

@custom_vjp
@jit
def flash_attention(q, k, v, mask):
    out, _ = flash_attn(q, k, v, mask)
    return out

@jit
def flash_attention_fwd(q, k, v, mask):
    out, lse = flash_attn(q, k, v, mask)
    return out, (q, k, v, mask, out, lse)

flash_attention.defvjp(flash_attention_fwd, flash_attention_bckwd)
