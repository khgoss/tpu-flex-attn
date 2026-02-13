import time
from flash_attention_jax import attention  # normal attn
from flash_impl import flash_attention_fwd  # custom flash
from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention as jax_flash_attention
from jax import random
import jax
import jax.numpy as jnp

print("JAX devices:", jax.devices())

# Benchmark settings
REPEAT = 30
WARMUP = 10
INPUT_SIZES = [1024, 2048, 4096]
BATCH = 2
HEADS = 8
HEAD_DIM = 64
SEQ_DIM = 512  # Total hidden dim = HEADS * HEAD_DIM = 8 * 64 = 512

def r(x):
    # (batch, seq_len, hidden) to (batch, heads, seq_len, head_dim)
    return jnp.reshape(x, (BATCH, HEADS, -1, HEAD_DIM))

def make_inputs(input_size, key):
    k1, k2, k3 = random.split(key, 3)
    q = r(random.normal(k1, (BATCH, input_size, SEQ_DIM)))
    k = r(random.normal(k2, (BATCH, input_size, SEQ_DIM)))
    v = r(random.normal(k3, (BATCH, input_size, SEQ_DIM)))
    mask = jnp.ones((BATCH, input_size), dtype=bool) # no mask!
    return q, k, v, mask

@jax.jit
def custom_flash(q, k, v, mask):
    out, _ = flash_attention_fwd(q, k, v, mask)
    return out

@jax.jit
def jax_flash(q, k, v, mask=None):
    return jax_flash_attention(q, k, v, causal=False)

@jax.jit
def vanilla(q, k, v, mask):
    return attention(q, k, v, mask)

print("\n" + "="*80)
print("WARMUP & JIT COMPILATION")
print("="*80)

# JIT compile and warmup with small input
dummy_q, dummy_k, dummy_v, dummy_mask = make_inputs(256, random.PRNGKey(0))

print("compiling custom flash")
_ = custom_flash(dummy_q, dummy_k, dummy_v, dummy_mask)
jax.block_until_ready(_)

print("compiling jax/pallas flash")
_ = jax_flash(dummy_q, dummy_k, dummy_v)
jax.block_until_ready(_)

print("compiling vanilla attn")
_ = vanilla(dummy_q, dummy_k, dummy_v, dummy_mask)
jax.block_until_ready(_)

print("warmup done!")


print("\n" + "="*80)
print("CORRECTNESS CHECK (seq_len=256)")
print("="*80)

test_q, test_k, test_v, test_mask = make_inputs(256, random.PRNGKey(42))

custom_out = custom_flash(test_q, test_k, test_v, test_mask)
jax_out = jax_flash(test_q, test_k, test_v)
vanilla_out = vanilla(test_q, test_k, test_v, test_mask)

custom_vs_vanilla = jnp.max(jnp.abs(custom_out - vanilla_out))
jax_vs_vanilla = jnp.max(jnp.abs(jax_out - vanilla_out))
custom_vs_jax = jnp.max(jnp.abs(custom_out - jax_out))

print(f"Custom Flash vs Vanilla:    max diff = {custom_vs_vanilla:.2e}")
print(f"JAX Flash vs Vanilla:       max diff = {jax_vs_vanilla:.2e}")
print(f"Custom Flash vs JAX Flash:  max diff = {custom_vs_jax:.2e}")


print(f"Batch={BATCH}, Heads={HEADS}, Head Dim={HEAD_DIM}, Repeat={REPEAT}")

results = {
    'custom': {},
    'jax': {},
    'vanilla': {}
}

print("\n" + "-"*80)
print("CUSTOM FLASH ATTENTION")
print("-"*80)
for input_size in INPUT_SIZES:
    times = []
    
    # warmup
    for _ in range(WARMUP):
        q, k, v, mask = make_inputs(input_size, random.PRNGKey(0))
        _ = custom_flash(q, k, v, mask)
        jax.block_until_ready(_)
    
    # benchmark
    for i in range(REPEAT):
        rng_key = random.PRNGKey(i)
        q, k, v, mask = make_inputs(input_size, rng_key)
        q, k, v, mask = jax.block_until_ready((q, k, v, mask))
        
        start = time.perf_counter()
        result = custom_flash(q, k, v, mask)
        jax.block_until_ready(result)
        times.append(time.perf_counter() - start)
    
    avg_time = sum(times) / REPEAT
    std_time = jnp.std(jnp.array(times))
    results['custom'][input_size] = avg_time
    print(f'Seq len: {input_size:4d} | Avg time: {avg_time*1000:7.3f} ms | Std: {std_time*1000:6.3f} ms')

# pallas flash
print("\n" + "-"*80)
print("JAX FLASH ATTENTION")
print("-"*80)
for input_size in INPUT_SIZES:
    times = []
    
    for _ in range(WARMUP):
        q, k, v, mask = make_inputs(input_size, random.PRNGKey(0))
        _ = jax_flash(q, k, v)
        jax.block_until_ready(_)
    
    for i in range(REPEAT):
        rng_key = random.PRNGKey(i)
        q, k, v, mask = make_inputs(input_size, rng_key)
        q, k, v = jax.block_until_ready((q, k, v))
        
        start = time.perf_counter()
        result = jax_flash(q, k, v)
        jax.block_until_ready(result)
        times.append(time.perf_counter() - start)
    
    avg_time = sum(times) / REPEAT
    std_time = jnp.std(jnp.array(times))
    results['jax'][input_size] = avg_time
    print(f'Seq len: {input_size:4d} | Avg time: {avg_time*1000:7.3f} ms | Std: {std_time*1000:6.3f} ms')


print("\n" + "-"*80)
print("VANILLA/DOT")
print("-"*80)
for input_size in INPUT_SIZES:
    times = []
    
    for _ in range(WARMUP):
        q, k, v, mask = make_inputs(input_size, random.PRNGKey(0))
        _ = vanilla(q, k, v, mask)
        jax.block_until_ready(_)
    
    for i in range(REPEAT):
        rng_key = random.PRNGKey(i)
        q, k, v, mask = make_inputs(input_size, rng_key)
        q, k, v, mask = jax.block_until_ready((q, k, v, mask))
        
        start = time.perf_counter()
        result = vanilla(q, k, v, mask)
        jax.block_until_ready(result)
        times.append(time.perf_counter() - start)
    
    avg_time = sum(times) / REPEAT
    std_time = jnp.std(jnp.array(times))
    results['vanilla'][input_size] = avg_time
    print(f'Seq len: {input_size:4d} | Avg time: {avg_time*1000:7.3f} ms | Std: {std_time*1000:6.3f} ms')


for input_size in INPUT_SIZES:
    custom_speedup = results['vanilla'][input_size] / results['custom'][input_size]
    jax_speedup = results['vanilla'][input_size] / results['jax'][input_size]
    custom_vs_jax = results['jax'][input_size] / results['custom'][input_size]
    
    print(f"{input_size:<10} {custom_speedup:>6.2f}x{'':<13} {jax_speedup:>6.2f}x{'':<13} "
          f"{'('+str(round(1/custom_vs_jax, 2))+'x slower)' if custom_vs_jax > 1 else '('+str(round(custom_vs_jax, 2))+'x faster)':<20}")


# for input_size in INPUT_SIZES:
#     custom_time = results['custom'][input_size] * 1000
#     jax_time = results['jax'][input_size] * 1000
#     vanilla_time = results['vanilla'][input_size] * 1000
    
#     print(f"{input_size:<10} {custom_time:>8.3f} ms{'':<5} {jax_time:>8.3f} ms{'':<5} {vanilla_time:>8.3f} ms")
