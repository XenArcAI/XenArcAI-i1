# -*- coding: utf-8 -*-
"""Xenith - A High-Performance Transformer Model
--------------------------------------------
Combines features for large context, robust MoE, efficient attention,
Hinglish support, and TPU optimization readiness.

Derived from concepts in requirements.txt, i12.txt, i1.txt, train.txt.
Version: 2.0.0 ("Beast") - Ready for Training Import
"""
import math
import os
import sys
import shutil # Keep for potential use in tokenizer creation/cleanup logic
import dataclasses # Keep for config saving/loading implementation
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import logging
import random

# --- Dependency Availability Checks & Imports ---

_XLA_AVAILABLE = False
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    _XLA_AVAILABLE = True
    logging.info("PyTorch XLA detected and imported successfully.")
except ImportError:
    logging.warning("PyTorch XLA not found. XLA-specific features will be disabled. Install for TPU usage.")
    # Define dummy functions/classes for compatibility when XLA is not available
    class xr:
        @staticmethod
        def world_size(): return 1
        @staticmethod
        def global_ordinal(): return 0
        # Note: is_xla_device check is now done via device.type == 'xla'
    class xm:
        @staticmethod
        def xla_device():
            if torch.cuda.is_available(): return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                 if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                      return torch.device("mps")
                 else: return torch.device("cpu") # Fallback if MPS not built/ready
            else: return torch.device("cpu")
        @staticmethod
        def save(*args, **kwargs): torch.save(*args, **kwargs) # Use torch.save if XLA not available

_SAFETENSORS_AVAILABLE = False
try:
    from safetensors.torch import save_file, load_file
    _SAFETENSORS_AVAILABLE = True
    logging.info("Safetensors detected and imported successfully.")
except ImportError:
    logging.warning("Safetensors not found. Saving/loading will use torch.save/load. Install safetensors for optimized saving.")

_TRANSFORMERS_AVAILABLE = False
try:
    from transformers import AutoTokenizer
    _TRANSFORMERS_AVAILABLE = True
    logging.info("Hugging Face Transformers detected and imported successfully.")
except ImportError:
    logging.error("Hugging Face Transformers library not found. This is required for tokenization. Please install it (`pip install transformers`).")
    # No point continuing without transformers
    sys.exit(1)

_INDICNLP_AVAILABLE = False
try:
    # Requires: pip install git+https://github.com/anoopkunchukuttan/indic_nlp_library.git pandas morfessor pyiwn==0.0.6
    from indicnlp.tokenize import indic_tokenize
    _INDICNLP_AVAILABLE = True
    logging.info("IndicNLP library detected and imported successfully.")
except ImportError:
    logging.warning("IndicNLP library not found or failed to import. Hindi token additions will be skipped. Ensure it's installed from its source repo if Hindi support is critical.")
    # Dummy class if not installed
    class indic_tokenize:
        @staticmethod
        def trivial_tokenize(text, lang='hi'): return []

try:
    import torch.utils.checkpoint as checkpoint
    _CHECKPOINT_AVAILABLE = True
except ImportError:
    logging.warning("torch.utils.checkpoint not available. Gradient checkpointing disabled.")
    _CHECKPOINT_AVAILABLE = False
    # Dummy checkpoint function
    def checkpoint(function, *args, **kwargs):
        # Simple passthrough if checkpointing isn't available
        # Ensure keyword arguments expected by the underlying function are handled if use_reentrant etc are passed.
        kwargs.pop('use_reentrant', None)
        kwargs.pop('preserve_rng_state', None)
        return function(*args, **kwargs) # Pass remaining kwargs

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s [%(name)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("XenithModel")

# --- Global XLA Settings ---
# These might be better handled in the training script, but keep for informational logging
world_size = 1
rank = 0
if _XLA_AVAILABLE:
    try:
        world_size = xr.world_size()
        rank = xr.global_ordinal()
        logger.info(f"XLA environment detected. World Size: {world_size}, Rank: {rank}")
    except Exception as e:
        logger.warning(f"Could not get XLA world size/rank via xr: {e}. Assuming single process.")
else:
     logger.info("XLA not available. Running in standard PyTorch mode.")

# --- Model Configuration ---
@dataclass
class ModelArgs:
    """Configuration arguments for the Xenith model."""
    # Core Architecture
    dim: int = 12288
    n_layers: int = 96
    n_heads: int = 128           # Query heads
    n_kv_heads: int = 32       # Key/Value heads (must divide n_heads for GQA)
    multiple_of: int = 4096     # Used for calculating hidden dim in MLP/MoE
    norm_eps: float = 1e-5

    # FeedForward Network (Dense Layers)
    ffn_dim_multiplier: Optional[float] = None # If set, overrides intermediate_dim calculation

    # Mixture of Experts (MoE)
    moe: bool = True
    n_routed_experts: int = 324  # Total number of experts per MoE layer
    n_activated_experts: int = 16 # Number of experts activated per token
    moe_intermediate_size: Optional[int] = None # If None, calculated based on dim/multiple_of
    shared_expert: bool = True # Whether to include a shared MLP alongside experts
    shared_expert_intermediate_size: Optional[int] = None # If None, calculated like MoE
    router_type: str = 'dynamic' # Options: 'dynamic' (learned gating), 'hash' (future), 'static' (future)
    router_bias: bool = False   # Bias in the router linear layer
    router_normalize_weights: bool = True # Normalize top-k weights to sum to 1
    router_temperature: float = 1.5 # Softmax temperature for routing (can be annealed)
    router_balance_weight: float = 0.05
    router_entropy_weight: float = 0.05 # Weight for encouraging router entropy (exploration)
    router_z_loss_weight: float = 1e-5 # Weight for Z-loss (prevents logit collapse)
    router_supervised_weight: float = 0.02 # Weight for optional supervised routing loss

    # LoRA Configuration
    use_lora: bool = True # Set to False if not using LoRA during pre-training/fine-tuning
    q_lora_rank: int = 512
    kv_lora_rank: int = 512
    o_lora_rank: int = 512      # LoRA rank for Attention output projection
    ffn_lora_rank: int = 512    # LoRA rank for FFN layers (MLP/Experts)

    # Sequence Length & Context Handling
    max_seq_len: int = 131072       # Max sequence length supported by RoPE precomputation
    rope_theta: float = 1000000.0   # RoPE base frequency
    rope_scaling_factor: float = 1.0 # YaRN scaling factor (1.0 = standard RoPE)
    rope_beta_fast: float = 32.0   # YaRN beta_fast parameter
    rope_beta_slow: float = 1.0    # YaRN beta_slow parameter (often 1)
    use_long_context_management: bool = True # Enable chunking/summarization
    context_chunk_size: int = 8192 # Size for chunking long inputs
    max_input_tokens_before_summarization: int = 100000 # Limit before chunking starts
    context_summary_size: int = 256 # Number of summary vectors per chunk
    context_memory_bank_max_chunks: int = 200 # Max summaries stored
    context_summarizer_n_layers: int = 8
    context_summarizer_n_heads: int = 16

    # Attention Configuration
    attention_dropout: float = 0.0 # Dropout in SDPA
    use_sparse_attention: bool = True # Use sparse attention for layers beyond the first few
    sparse_attention_window_size: int = 2048
    sparse_attention_global_tokens: int = 256
    sparse_attention_dynamic: bool = True # Use policy network to adjust window/global sizes
    use_full_attention_first_n_layers: int = 2 # Number of initial layers using full attention

    # Other Training/Model Parameters
    vocab_size: int = -1           # CRITICAL: Must be set by tokenizer initialization
    dropout: float = 0.05          # General dropout for non-attention layers
    use_gradient_checkpointing: bool = True # Enable gradient checkpointing (conditional on non-XLA)
    use_swiglu: bool = True        # Use SwiGLU activation in FFNs
    use_layer_norm: bool = False   # Use LayerNorm instead of RMSNorm
    n_dense_layers: int = 6       # Number of initial layers using dense MLP instead of MoE
    rl_reward_correct: float = 10.0 # Reward for 'correct' routing/summarization actions
    rl_reward_incorrect: float = -10.0 # Penalty for 'incorrect' actions
    combined_tokenizer_path: str = "xenith_combined_tokenizer" # Default path to load/save the combined tokenizer. CHANGE THIS for your setup.
    model_dtype: torch.dtype = torch.bfloat16 # Default dtype (ensure hardware supports it)

    # Calculated fields (initialized in __post_init__)
    head_dim: int = field(init=False)
    intermediate_dim: int = field(init=False) # For dense MLP layers
    moe_final_intermediate_size: int = field(init=False) # For MoE expert layers
    shared_final_intermediate_size: int = field(init=False) # For shared expert

    def __post_init__(self):
        if self.dim % self.n_heads != 0:
            raise ValueError(f"Model dim {self.dim} must be divisible by n_heads {self.n_heads}")
        self.head_dim = self.dim // self.n_heads

        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(f"n_heads {self.n_heads} must be divisible by n_kv_heads {self.n_kv_heads}")

        if self.vocab_size <= 0:
             logger.warning("ModelArgs.vocab_size not set during init. It MUST be updated by tokenizer.")

        # Calculate intermediate dimensions for FFNs
        hidden_dim = self.dim * 4 # Standard heuristic
        hidden_dim = int(2 * hidden_dim / 3) # LLaMA heuristic
        # Ensure hidden_dim is a multiple of multiple_of for efficiency
        self.intermediate_dim = self.multiple_of * ((hidden_dim + self.multiple_of - 1) // self.multiple_of)
        if self.ffn_dim_multiplier is not None:
            self.intermediate_dim = int(self.ffn_dim_multiplier * self.dim)
            self.intermediate_dim = self.multiple_of * ((self.intermediate_dim + self.multiple_of - 1) // self.multiple_of)

        # Calculate intermediate dimensions for MoE experts
        if self.moe_intermediate_size is None:
            moe_hidden_dim = int(self.dim * 8 / 3) # Example: Slightly smaller than 4x
            moe_hidden_dim = int(2 * moe_hidden_dim / 3)
            self.moe_final_intermediate_size = self.multiple_of * ((moe_hidden_dim + self.multiple_of - 1) // self.multiple_of)
        else:
            self.moe_final_intermediate_size = self.moe_intermediate_size

        # Calculate intermediate dimensions for Shared Expert (if enabled)
        if self.shared_expert_intermediate_size is None:
             self.shared_final_intermediate_size = self.moe_final_intermediate_size
        else:
             self.shared_final_intermediate_size = self.shared_expert_intermediate_size

        logger.info(f"Calculated Dimensions: head_dim={self.head_dim}, intermediate_dim={self.intermediate_dim}, moe_intermediate_size={self.moe_final_intermediate_size}")


# --- Specialized Layers ---

class ParallelEmbedding(nn.Module):
    """Token embedding layer."""
    def __init__(self, vocab_size: int, dim: int, dtype: torch.dtype):
        super().__init__()
        if vocab_size <= 0: raise ValueError("vocab_size must be positive for Embedding.")
        self.vocab_size = vocab_size
        self.dim = dim
        self.weight = nn.Parameter(torch.empty(vocab_size, dim, dtype=dtype))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        logger.debug(f"Initialized ParallelEmbedding: vocab={vocab_size}, dim={dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.embedding(x, self.weight)

class LoRALinear(nn.Module):
    """Linear layer with optional LoRA adaptation."""
    def __init__(self, in_features: int, out_features: int, lora_rank: int, bias: bool = False, use_lora: bool = True, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank if use_lora else 0
        self.use_lora = use_lora

        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype), requires_grad=not (self.use_lora and self.lora_rank > 0))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

        if self.use_lora and self.lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(lora_rank, in_features, dtype=dtype))
            self.lora_B = nn.Parameter(torch.empty(out_features, lora_rank, dtype=dtype))
            nn.init.zeros_(self.lora_A)
            nn.init.normal_(self.lora_B, std=(1 / lora_rank))
            # logger.debug(f"Initialized LoRALinear: in={in_features}, out={out_features}, rank={lora_rank}, bias={bias}") # Reduced logging verbosity
        else:
            self.register_parameter('lora_A', None)
            self.register_parameter('lora_B', None)
            # logger.debug(f"Initialized Linear (No LoRA): in={in_features}, out={out_features}, bias={bias}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        if self.use_lora and self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            x_dtype = x.dtype
            lora_adj = F.linear(F.linear(x.to(self.lora_A.dtype), self.lora_A), self.lora_B)
            y = y + lora_adj.to(x_dtype)
        return y

class RMSNorm(nn.Module):
    """Root Mean Square Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))
        self.eps = eps
        # logger.debug(f"Initialized RMSNorm: dim={dim}, eps={eps}")

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).to(x.dtype)
        return output * self.weight

class LayerNorm(nn.Module):
    """Standard Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(dim, dtype=dtype))
        self.eps = eps
        # logger.debug(f"Initialized LayerNorm: dim={dim}, eps={eps}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (x.size(-1),), self.weight, self.bias, self.eps)

# --- RoPE and Attention Classes (Using SDPA) ---

def _yarn_find_correction_dim(num_rotations, dim, base=10000.0, max_position_embeddings=2048):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

def _yarn_find_correction_range(low_rot, high_rot, dim, base=10000.0, max_position_embeddings=2048):
    low = math.floor(_yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(_yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(0, low), min(dim - 1, high)

def _yarn_linear_ramp_mask(min, max, dim):
    if min == max: max += 0.001
    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func

def precompute_freqs_cis_yarn(dim: int, end: int, theta: float = 10000.0,
                              scaling_factor: float = 1.0, beta_fast: float = 32.0, beta_slow: float = 1.0,
                              original_max_position_embeddings: int = 4096,
                              dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Precomputes the rotary frequency tensor with YaRN scaling."""
    inv_freq = 1.0 / (theta**(torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if scaling_factor != 1.0:
        scale = scaling_factor
        max_pos_emb = original_max_position_embeddings
        low, high = _yarn_find_correction_range(beta_fast, beta_slow, dim, theta, max_pos_emb)
        inv_freq_mask = (1 - _yarn_linear_ramp_mask(low, high, dim // 2)).float()
        inv_freq = inv_freq * (inv_freq_mask * scale + (1.0 - inv_freq_mask))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    # logger.debug(f"Computed RoPE freqs cis: shape={freqs_cis.shape}, dtype={freqs_cis.dtype}")
    return freqs_cis.to(dtype=torch.complex64)

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Applies RoPE embeddings to input tensor x."""
    x_dtype = x.dtype
    head_dim = x.shape[-1]
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    if freqs_cis.dim() == 2:
        freqs_cis = freqs_cis.unsqueeze(1)
    freqs_cis = freqs_cis.unsqueeze(0)
    freqs_cis = freqs_cis.to(device=x_complex.device, dtype=x_complex.dtype)
    x_rotated_complex = x_complex * freqs_cis
    x_rotated_real = torch.view_as_real(x_rotated_complex)
    x_out = x_rotated_real.flatten(start_dim=-2)
    return x_out.to(x_dtype)


class BaseAttention(nn.Module):
    """Base Attention class using SDPA, handling GQA and KV caching."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.head_dim
        self.dim = args.dim
        self.dtype = args.model_dtype
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})")
        self.num_key_value_groups = self.n_heads // self.n_kv_heads
        self.wq = LoRALinear(self.dim, self.n_heads * self.head_dim, args.q_lora_rank, bias=False, use_lora=args.use_lora, dtype=self.dtype)
        self.wk = LoRALinear(self.dim, self.n_kv_heads * self.head_dim, args.kv_lora_rank, bias=False, use_lora=args.use_lora, dtype=self.dtype)
        self.wv = LoRALinear(self.dim, self.n_kv_heads * self.head_dim, args.kv_lora_rank, bias=False, use_lora=args.use_lora, dtype=self.dtype)
        self.wo = LoRALinear(self.n_heads * self.head_dim, self.dim, args.o_lora_rank, bias=False, use_lora=args.use_lora, dtype=self.dtype)
        self.attention_dropout = args.attention_dropout
        self.kv_cache = None
        self.register_buffer("freqs_cis", precompute_freqs_cis_yarn(
            self.head_dim, args.max_seq_len, args.rope_theta,
            args.rope_scaling_factor, args.rope_beta_fast, args.rope_beta_slow,
            args.max_seq_len // int(args.rope_scaling_factor or 1.0),
            dtype=self.dtype
            ), persistent=False)

    def _reset_kv_cache(self):
        self.kv_cache = None

    def _update_kv_cache(self, k_new: torch.Tensor, v_new: torch.Tensor, start_pos: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if start_pos == 0:
            self.kv_cache = (k_new, v_new)
        else:
            if self.kv_cache is None:
                raise RuntimeError("KV cache is None but start_pos > 0.")
            k_cached, v_cached = self.kv_cache
            if k_cached.shape[0]!=k_new.shape[0] or k_cached.shape[2]!=k_new.shape[2] or k_cached.shape[3]!=k_new.shape[3]:
                 raise ValueError(f"KV cache shape mismatch: Cached K ({k_cached.shape}), New K ({k_new.shape})")
            if v_cached.shape[0]!=v_new.shape[0] or v_cached.shape[2]!=v_new.shape[2] or v_cached.shape[3]!=v_new.shape[3]:
                 raise ValueError(f"KV cache shape mismatch: Cached V ({v_cached.shape}), New V ({v_new.shape})")
            k_updated = torch.cat([k_cached, k_new], dim=1)
            v_updated = torch.cat([v_cached, v_new], dim=1)
            self.kv_cache = (k_updated, v_updated)
        return self.kv_cache

    def _repeat_kv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, n_kv_heads, head_dim = hidden_states.shape
        if self.num_key_value_groups == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, :, None, :].expand(bsz, seq_len, n_kv_heads, self.num_key_value_groups, head_dim)
        return hidden_states.reshape(bsz, seq_len, n_kv_heads * self.num_key_value_groups, head_dim)

    def _apply_rope(self, query: torch.Tensor, key: torch.Tensor, start_pos: int) -> Tuple[torch.Tensor, torch.Tensor]:
         seq_len = query.shape[1]
         freqs_cis_current = self.freqs_cis[start_pos : start_pos + seq_len]
         query_rotated = apply_rotary_emb(query, freqs_cis_current)
         key_rotated = apply_rotary_emb(key, freqs_cis_current)
         return query_rotated, key_rotated

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor], start_pos: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, q_len, _ = x.shape
        query_states = self.wq(x).view(bsz, q_len, self.n_heads, self.head_dim)
        key_states = self.wk(x).view(bsz, q_len, self.n_kv_heads, self.head_dim)
        value_states = self.wv(x).view(bsz, q_len, self.n_kv_heads, self.head_dim)
        query_states, key_states = self._apply_rope(query_states, key_states, start_pos)
        k_cache, v_cache = self._update_kv_cache(key_states, value_states, start_pos)
        cache_seq_len = k_cache.shape[1]
        key_states_rep = self._repeat_kv(k_cache)
        value_states_rep = self._repeat_kv(v_cache)
        query_states = query_states.transpose(1, 2)
        key_states_rep = key_states_rep.transpose(1, 2)
        value_states_rep = value_states_rep.transpose(1, 2)

        sdpa_mask_bool = None
        if attention_mask is not None:
            if attention_mask.dtype != torch.bool:
                 current_mask = attention_mask[:, :, -q_len:, :cache_seq_len]
                 sdpa_mask_bool = (current_mask <= -1e9)
            else:
                 sdpa_mask_bool = attention_mask[:, :, -q_len:, :cache_seq_len]
            # Simple check, SDPA handles most broadcasts
            if sdpa_mask_bool.dim() != 4 and sdpa_mask_bool.dim() != 2:
                 logger.warning(f"Unexpected mask dimension: {sdpa_mask_bool.dim()}. Expected 2 or 4.")
                 sdpa_mask_bool = None # Fallback

        attn_output = F.scaled_dot_product_attention(
            query_states, key_states_rep, value_states_rep,
            attn_mask=sdpa_mask_bool,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.dim)
        output = self.wo(attn_output)
        attn_reg_loss = torch.tensor(0.0, device=output.device, dtype=output.dtype)
        attn_reward = torch.tensor(0.0, device=output.device, dtype=output.dtype)
        return output, attn_reg_loss, attn_reward


class FullAttention(BaseAttention):
    """Standard Full Attention using the BaseAttention class."""
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor], start_pos: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return super().forward(x, attention_mask, start_pos)

class SparseAttention(BaseAttention):
    """Sparse Attention using the BaseAttention class with a dynamic sparse mask."""
    def __init__(self, args: ModelArgs):
        super().__init__(args)
        self.window_size = args.sparse_attention_window_size
        self.global_tokens = args.sparse_attention_global_tokens
        self.use_dynamic = args.sparse_attention_dynamic
        if self.use_dynamic:
            self.policy_network = nn.Sequential(
                nn.Linear(args.dim, 256, dtype=args.model_dtype, bias=False),
                nn.ReLU(),
                nn.Linear(256, 2, dtype=args.model_dtype, bias=False),
                nn.Tanh()
            )
            # logger.info("Initialized Dynamic Sparse Attention Policy Network.")
        else:
            self.policy_network = None

    def _get_sparse_attn_mask(self, x: torch.Tensor, q_len: int, k_len: int) -> torch.Tensor:
        """Generates an ADDITIVE attention mask for sparse attention (-inf for masked)."""
        bsz = x.size(0)
        device = x.device
        dtype = self.args.model_dtype
        if self.use_dynamic and self.policy_network is not None:
            with torch.no_grad():
                context_signal = x.mean(dim=1).to(self.policy_network[0].weight.dtype)
                adjustments = self.policy_network(context_signal)
                window_adjust = adjustments[:, 0] * (self.window_size * 0.5)
                global_adjust = adjustments[:, 1] * (self.global_tokens * 0.5)
                dyn_ws = (torch.full((bsz,), self.window_size, device=device) + window_adjust).clamp(min=64, max=self.window_size * 2).int()
                dyn_gt = (torch.full((bsz,), self.global_tokens, device=device) + global_adjust).clamp(min=16, max=self.global_tokens * 2).int()
        else:
            dyn_ws = torch.full((bsz,), self.window_size, device=device, dtype=torch.int)
            dyn_gt = torch.full((bsz,), self.global_tokens, device=device, dtype=torch.int)

        max_global = dyn_gt.max().item()
        global_indices = torch.arange(max_global, device=device)
        global_attend_mask = (global_indices.view(1, 1, -1) < dyn_gt.view(-1, 1, 1))
        global_mask = F.pad(global_attend_mask, (0, k_len - max_global), value=False)
        query_indices = torch.arange(q_len, device=device).view(1, q_len, 1)
        key_indices = torch.arange(k_len, device=device).view(1, 1, k_len)
        half_window = dyn_ws.view(-1, 1, 1) // 2
        lower_bound = query_indices - half_window
        upper_bound = query_indices + half_window
        local_mask = (key_indices >= lower_bound) & (key_indices <= upper_bound)
        combined_mask = global_mask | local_mask
        additive_mask = torch.zeros(bsz, 1, q_len, k_len, device=device, dtype=dtype)
        additive_mask.masked_fill_(~combined_mask.unsqueeze(1), float("-inf"))
        return additive_mask

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor], start_pos: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_len = x.size(1)
        k_len = self.kv_cache[0].size(1) if self.kv_cache is not None and start_pos > 0 else q_len
        sparse_mask = self._get_sparse_attn_mask(x, q_len, k_len)
        if attention_mask is not None:
            input_mask_part = attention_mask[:, :, -q_len:, :k_len]
            combined_mask = torch.clamp(input_mask_part + sparse_mask, max=0.0)
        else:
            combined_mask = sparse_mask
        return super().forward(x, combined_mask, start_pos)


# --- Long Context Management ---

class TransformerSummarizer(nn.Module):
    """Summarizes input sequences using a small Transformer encoder and attention pooling."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.dim = args.dim
        self.summary_size = args.context_summary_size
        self.dtype = args.model_dtype
        if args.context_summarizer_n_layers <= 0:
             logger.warning("TransformerSummarizer disabled (n_layers <= 0). Returning zeros.")
             self.encoder = None; self.attention_pool = None; self.summary_projector = None; self.summary_query = None
        else:
             encoder_layer = nn.TransformerEncoderLayer(
                 d_model=self.dim, nhead=args.context_summarizer_n_heads, dim_feedforward=self.dim * 4,
                 dropout=args.dropout, activation=F.gelu, batch_first=True, dtype=self.dtype, norm_first=True
             )
             self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=args.context_summarizer_n_layers)
             self.summary_query = nn.Parameter(torch.randn(1, self.summary_size, self.dim, dtype=self.dtype))
             nn.init.normal_(self.summary_query, std=0.02)
             self.attention_pool = nn.MultiheadAttention(
                 embed_dim=self.dim, num_heads=args.context_summarizer_n_heads, dropout=args.dropout, batch_first=True, dtype=self.dtype
             )
             self.summary_projector = LoRALinear(
                 self.dim, self.dim, args.ffn_lora_rank, bias=False, use_lora=args.use_lora, dtype=self.dtype
             )
             # logger.info(f"Initialized TransformerSummarizer: layers={args.context_summarizer_n_layers}, heads={args.context_summarizer_n_heads}, summary_size={self.summary_size}")

    def _compute_summary_reward(self, summary: torch.Tensor, original_input: torch.Tensor) -> torch.Tensor:
        if summary.numel() == 0 or original_input.numel() == 0: return torch.tensor(0.0, device=summary.device, dtype=self.dtype)
        summary_repr = summary.mean(dim=1).float()
        input_repr = original_input.mean(dim=1).float()
        similarity = F.cosine_similarity(summary_repr, input_repr, dim=-1).mean()
        reward = torch.where(similarity > 0.7,
                           torch.tensor(self.args.rl_reward_correct, dtype=torch.float, device=summary.device),
                           torch.tensor(self.args.rl_reward_incorrect, dtype=torch.float, device=summary.device))
        return reward.to(self.dtype)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = x.size(0)
        if self.encoder is None:
             dummy_summary = torch.zeros(bsz, self.summary_size, self.dim, device=x.device, dtype=self.dtype)
             dummy_reward = torch.tensor(0.0, device=x.device, dtype=self.dtype)
             return dummy_summary, dummy_reward
        encoded_x = self.encoder(x)
        query = self.summary_query.expand(bsz, -1, -1)
        summary, _ = self.attention_pool(query=query, key=encoded_x, value=encoded_x)
        if self.summary_projector: summary = self.summary_projector(summary)
        reward = self._compute_summary_reward(summary.detach(), x.detach())
        return summary, reward

class DynamicMemoryBank:
    """Stores and retrieves summaries using an OrderedDict as an LRU cache."""
    def __init__(self, dim: int, summary_size: int, max_chunks: int, dtype: torch.dtype):
        self.memory: OrderedDict[int, torch.Tensor] = OrderedDict()
        self.max_chunks = max_chunks
        self.summary_size = summary_size
        self.dim = dim
        self.persistent_keys: set[int] = set()
        self.dtype = dtype
        logger.info(f"Initialized DynamicMemoryBank: max_chunks={max_chunks}, summary_size={summary_size}, dim={dim}")

    def add(self, summary: torch.Tensor, key: int, persistent: bool = False):
        if summary.dim() != 2 or summary.shape[0] != self.summary_size or summary.shape[1] != self.dim:
             logger.error(f"Invalid summary shape for adding: {summary.shape}. Expected ({self.summary_size}, {self.dim})")
             return
        summary_cpu = summary.detach().to('cpu', non_blocking=True)
        if key in self.memory: self.memory.pop(key); self.persistent_keys.discard(key)
        if len(self.memory) >= self.max_chunks and key not in self.persistent_keys:
            key_to_evict = next((k for k in self.memory if k not in self.persistent_keys), -1)
            if key_to_evict != -1: self.memory.pop(key_to_evict); logger.debug(f"Mem bank evicted {key_to_evict}")
            else: logger.warning(f"Mem bank full persistent ({len(self.memory)}/{self.max_chunks}). Cannot add {key}."); return
        self.memory[key] = summary_cpu; self.memory.move_to_end(key)
        if persistent: self.persistent_keys.add(key)

    def retrieve(self, query: torch.Tensor, k: int = 5) -> Optional[torch.Tensor]:
        if not self.memory: return None
        bsz = query.size(0); device = query.device; query_dtype = query.dtype
        if query.dim() != 3 or query.shape[-1] != self.dim:
             logger.error(f"Invalid query shape: {query.shape}. Expected (bsz, seq_len, {self.dim})"); return None
        query_vector = query.mean(dim=1).float()
        try: summaries = torch.stack([s.to(device=device, dtype=query_dtype) for s in self.memory.values()])
        except Exception as e: logger.error(f"Error stacking summaries: {e}"); return torch.zeros(bsz, self.summary_size, self.dim, device=device, dtype=query_dtype)
        if summaries.numel() == 0: return None
        summary_vectors = summaries.mean(dim=1).float()
        similarity = F.cosine_similarity(query_vector.unsqueeze(1), summary_vectors.unsqueeze(0), dim=-1)
        num_available = summaries.size(0); actual_k = min(k, num_available)
        top_k_scores, top_k_indices = torch.topk(similarity, actual_k, dim=-1)
        retrieved_summaries = summaries[top_k_indices]
        weights = F.softmax(top_k_scores, dim=-1).to(query_dtype).unsqueeze(-1).unsqueeze(-1)
        weighted_memory = (retrieved_summaries * weights).sum(dim=1)
        return weighted_memory.to(query_dtype)

    def get_memory_usage_mb(self) -> float:
        if not self.memory: return 0.0
        # Estimate element size robustly
        try: element_size = torch.tensor([], dtype=self.dtype).element_size()
        except: element_size = 4 # Default to float32 if dtype unknown or causes error
        entry_size_bytes = self.summary_size * self.dim * element_size
        total_bytes = len(self.memory) * entry_size_bytes
        return total_bytes / (1024 * 1024)

    def clear(self):
        self.memory.clear(); self.persistent_keys.clear()
        logger.info("DynamicMemoryBank cleared.")

# --- FeedForward and MoE Layers ---
class MLP(nn.Module):
    """Standard FeedForward Network (MLP) with optional SwiGLU."""
    def __init__(self, args: ModelArgs, intermediate_size: int, is_shared_expert: bool = False):
        super().__init__()
        self.dim = args.dim; self.intermediate_size = intermediate_size; self.use_swiglu = args.use_swiglu; self.dtype = args.model_dtype
        lora_rank = args.ffn_lora_rank if args.use_lora else 0
        if self.use_swiglu:
            self.w1 = LoRALinear(self.dim, self.intermediate_size, lora_rank, bias=False, use_lora=args.use_lora, dtype=self.dtype)
            self.w3 = LoRALinear(self.dim, self.intermediate_size, lora_rank, bias=False, use_lora=args.use_lora, dtype=self.dtype)
            self.w2 = LoRALinear(self.intermediate_size, self.dim, lora_rank, bias=False, use_lora=args.use_lora, dtype=self.dtype)
            self.activation_fn = F.silu
        else:
            self.w1 = LoRALinear(self.dim, self.intermediate_size, lora_rank, bias=False, use_lora=args.use_lora, dtype=self.dtype)
            self.w2 = LoRALinear(self.intermediate_size, self.dim, lora_rank, bias=False, use_lora=args.use_lora, dtype=self.dtype)
            self.register_parameter('w3', None); self.activation_fn = F.gelu
        self.dropout = nn.Dropout(args.dropout)
        # layer_type = "SharedExpertMLP" if is_shared_expert else "DenseMLP" # Reduced logging
        # logger.debug(f"Initialized {layer_type}: dim={self.dim}, intermediate={self.intermediate_size}, swiglu={self.use_swiglu}, lora_rank={lora_rank}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_swiglu:
             if hasattr(self, 'w3') and self.w3 is not None: hidden_states = self.activation_fn(self.w1(x)) * self.w3(x)
             else: logger.error("SwiGLU enabled but w3 layer missing."); hidden_states = self.activation_fn(self.w1(x))
        else: hidden_states = self.activation_fn(self.w1(x))
        hidden_states = self.dropout(hidden_states); output = self.w2(hidden_states); return output

class Expert(MLP):
     """Expert module, essentially an MLP."""
     def __init__(self, args: ModelArgs):
         super().__init__(args, args.moe_final_intermediate_size)
         # logger.debug(f"Initialized Expert: dim={self.dim}, intermediate={self.intermediate_size}, swiglu={self.use_swiglu}")

class DynamicRouter(nn.Module):
    """Routes tokens to experts based on learned scores."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args; self.n_experts = args.n_routed_experts; self.top_k = args.n_activated_experts; self.dtype = args.model_dtype
        self.scorer = nn.Linear(args.dim, self.n_experts, bias=args.router_bias, dtype=self.dtype); nn.init.zeros_(self.scorer.weight)
        if args.router_bias: nn.init.zeros_(self.scorer.bias)
        self.temperature = nn.Parameter(torch.tensor(args.router_temperature, dtype=torch.float), requires_grad=False)
        self.balance_weight = args.router_balance_weight; self.entropy_weight = args.router_entropy_weight; self.z_loss_weight = args.router_z_loss_weight
        self.supervised_weight = args.router_supervised_weight; self.normalize_weights = args.router_normalize_weights
        self.step = 0; self.total_steps_for_anneal = 100000 # Example
        self.dropout = nn.Dropout(args.dropout)
        # logger.info(f"Initialized DynamicRouter: n_experts={self.n_experts}, top_k={self.top_k}, temp={self.temperature.item()}")

    def _anneal_temperature(self):
        if self.training:
            progress = min(1.0, self.step / self.total_steps_for_anneal)
            # Linear annealing from 1.5 to 0.5
            initial_temp = 1.5  # Starting value
            final_temp = 0.5   # Target value
            new_temp = initial_temp - (initial_temp - final_temp) * progress  # Linear decay
            new_temp = max(final_temp, new_temp)  # Ensure it doesn't go below 0.5
            self.temperature.data = torch.tensor(new_temp, device=self.temperature.device, dtype=torch.float)
            self.step += 1

    def _compute_router_reward(self, chosen_indices: torch.Tensor, expert_labels: torch.Tensor) -> torch.Tensor:
        if expert_labels is None: return torch.tensor(0.0, device=chosen_indices.device, dtype=self.dtype)
        valid_mask = (expert_labels != -100)
        if not valid_mask.any(): return torch.tensor(0.0, device=chosen_indices.device, dtype=self.dtype)
        valid_indices = chosen_indices[valid_mask]; valid_labels = expert_labels[valid_mask].long()
        correct_routing = torch.any(valid_indices == valid_labels.unsqueeze(-1), dim=-1)
        reward = torch.where(correct_routing,
                             torch.tensor(self.args.rl_reward_correct, dtype=torch.float, device=chosen_indices.device),
                             torch.tensor(self.args.rl_reward_incorrect, dtype=torch.float, device=chosen_indices.device))
        return reward.mean().to(self.dtype)

    def forward(self, x: torch.Tensor, expert_labels: Optional[torch.Tensor] = None) -> \
                 Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_tokens, _ = x.shape; x_float = x.float()
        self._anneal_temperature()
        router_logits = self.scorer(x_float)
        router_probs = F.softmax(router_logits / self.temperature.clamp(min=0.1), dim=-1)
        router_probs = self.dropout(router_probs)
        top_k_weights, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        final_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-9) if self.normalize_weights else top_k_weights
        expert_mask = F.one_hot(top_k_indices, num_classes=self.n_experts).float()
        tokens_per_expert = torch.einsum('tk,tke->e', final_weights.float(), expert_mask)
        expert_usage_fraction = tokens_per_expert / (num_tokens + 1e-9)
        load_variance_loss = expert_usage_fraction.var()
        balance_loss = self.balance_weight * load_variance_loss
        entropy = -(router_probs * torch.log(router_probs + 1e-9)).sum(dim=-1).mean()
        entropy_loss = -self.entropy_weight * entropy
        log_z = torch.logsumexp(router_logits, dim=-1)
        z_loss = self.z_loss_weight * (log_z**2).mean()
        total_aux_loss = balance_loss + entropy_loss + z_loss
        supervised_loss = torch.tensor(0.0, device=x.device, dtype=self.dtype)
        if expert_labels is not None and self.supervised_weight > 0:
            valid_mask = (expert_labels != -100)
            if valid_mask.any():
                 supervised_loss = F.cross_entropy(router_logits[valid_mask], expert_labels[valid_mask].long(), reduction='mean')
                 supervised_loss = supervised_loss * self.supervised_weight
        router_reward = self._compute_router_reward(top_k_indices.detach(), expert_labels)
        return (final_weights.to(self.dtype), top_k_indices, expert_usage_fraction.float(),
                total_aux_loss.to(self.dtype), supervised_loss.to(self.dtype), router_reward.to(self.dtype))

class MoE(nn.Module):
    """Mixture of Experts layer."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args; self.dim = args.dim; self.n_experts = args.n_routed_experts; self.top_k = args.n_activated_experts; self.dtype = args.model_dtype
        if args.router_type == 'dynamic': self.router = DynamicRouter(args)
        else: raise NotImplementedError(f"Router type '{args.router_type}' not implemented.")
        self.experts = nn.ModuleList([Expert(args) for _ in range(self.n_experts)])
        if args.shared_expert:
            self.shared_expert_mlp = MLP(args, args.shared_final_intermediate_size, is_shared_expert=True)
            # logger.info("Initialized MoE layer with Shared Expert MLP.")
        else: self.register_parameter('shared_expert_mlp', None)

    def forward(self, x: torch.Tensor, expert_labels: Optional[torch.Tensor] = None) -> \
                 Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, dim = x.shape; x_flat = x.view(-1, dim)
        flat_labels = expert_labels.view(-1) if expert_labels is not None else None
        router_weights, expert_indices, usage, balance_loss, supervised_loss, router_reward = self.router(x_flat, flat_labels)
        final_output_flat = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            token_indices, top_k_idx = torch.where(expert_indices == i)
            if token_indices.numel() > 0:
                expert_input = x_flat[token_indices]
                expert_output = expert(expert_input)
                weights_for_expert = router_weights[token_indices, top_k_idx]
                final_output_flat.index_add_(0, token_indices, expert_output * weights_for_expert.unsqueeze(-1))
        if self.shared_expert_mlp is not None:
            shared_output = self.shared_expert_mlp(x_flat)
            final_output_flat = final_output_flat + shared_output
        output = final_output_flat.view(bsz, seq_len, dim)
        return output, usage, balance_loss, supervised_loss, router_reward

class LanguageAdapter(nn.Module):
    """Simple adapter module for language-specific adjustments."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim; self.lora_rank = args.ffn_lora_rank if args.use_lora else 0; self.use_lora = args.use_lora; self.dtype = args.model_dtype
        bottleneck_dim = self.dim // 4
        self.down_proj = LoRALinear(self.dim, bottleneck_dim, self.lora_rank, bias=False, use_lora=self.use_lora, dtype=self.dtype)
        self.up_proj = LoRALinear(bottleneck_dim, self.dim, self.lora_rank, bias=False, use_lora=self.use_lora, dtype=self.dtype)
        self.activation = nn.GELU(); self.dropout = nn.Dropout(args.dropout)
        # logger.debug(f"Initialized LanguageAdapter: dim={self.dim}, bottleneck={bottleneck_dim}, lora_rank={self.lora_rank}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.down_proj(x); hidden = self.activation(hidden); hidden = self.dropout(hidden); output = self.up_proj(hidden); return output

# --- Transformer Block ---

class Block(nn.Module):
    """A single Transformer block, combining attention and FFN/MoE."""
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id; self.args = args; self.dim = args.dim; self.dtype = args.model_dtype; self.gradient_checkpointing = args.use_gradient_checkpointing
        use_full_attn = layer_id < args.use_full_attention_first_n_layers
        if use_full_attn: self.attention = FullAttention(args)
        elif args.use_sparse_attention: self.attention = SparseAttention(args)
        else: self.attention = FullAttention(args)
        self.is_moe_layer = args.moe and layer_id >= args.n_dense_layers
        if self.is_moe_layer: self.feed_forward = MoE(args)
        else: self.feed_forward = MLP(args, args.intermediate_dim)
        norm_class = LayerNorm if args.use_layer_norm else RMSNorm
        self.attention_norm = norm_class(args.dim, eps=args.norm_eps, dtype=self.dtype)
        self.ffn_norm = norm_class(args.dim, eps=args.norm_eps, dtype=self.dtype)
        self.dropout = nn.Dropout(args.dropout)
        if layer_id % 2 == 0: self.language_adapter = LanguageAdapter(args)
        else: self.register_parameter('language_adapter', None)
        if args.use_long_context_management:
             self.memory_gate = nn.Linear(args.dim * 2, args.dim, bias=False, dtype=self.dtype)
             self.memory_gate_activation = nn.Sigmoid()
        else: self.register_parameter('memory_gate', None)
        self.is_on_xla = None

    def _forward_impl(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor], memory_context: Optional[torch.Tensor],
                      expert_labels: Optional[torch.Tensor], languages: Optional[List[str]], start_pos: int
                      ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = x; attn_input = x
        if memory_context is not None and hasattr(self, 'memory_gate') and self.memory_gate is not None:
            if memory_context.size(1) == 1: memory_expanded = memory_context.expand(-1, x.size(1), -1)
            elif memory_context.size(1) == x.size(1): memory_expanded = memory_context
            else:
                logger.warning(f"Layer {self.layer_id}: Memory context shape {memory_context.shape} != input shape {x.shape}. Using mean.")
                memory_expanded = memory_context.mean(dim=1, keepdim=True).expand(-1, x.size(1), -1)
            combined_input_memory = torch.cat([x, memory_expanded], dim=-1)
            gate_values = self.memory_gate_activation(self.memory_gate(combined_input_memory.to(self.memory_gate.weight.dtype))).to(x.dtype)
            attn_input = x + gate_values * memory_expanded
        hidden_states_norm = self.attention_norm(attn_input)
        attn_output, attn_reg_loss, attn_reward = self.attention(hidden_states_norm, attention_mask, start_pos)
        hidden_states = residual + self.dropout(attn_output)
        if self.language_adapter is not None and languages is not None:
             bsz = hidden_states.size(0)
             if len(languages) == bsz:
                 lang_mask = torch.tensor([1.0 if lang.lower() in ['hindi', 'hin', 'hi', 'hinglish'] else 0.0 for lang in languages],
                                          device=hidden_states.device, dtype=hidden_states.dtype).view(bsz, 1, 1)
                 adapter_output = self.language_adapter(hidden_states)
                 hidden_states = hidden_states + lang_mask * adapter_output
             else: logger.warning(f"Layer {self.layer_id}: Batch size ({bsz}) != languages provided ({len(languages)}). Skipping adapter.")
        ffn_input_norm = self.ffn_norm(hidden_states)
        moe_usage, moe_balance_loss, moe_supervised_loss, moe_router_reward = None, torch.zeros_like(attn_reg_loss), torch.zeros_like(attn_reg_loss), torch.zeros_like(attn_reward)
        if self.is_moe_layer: ffn_output, moe_usage, moe_balance_loss, moe_supervised_loss, moe_router_reward = self.feed_forward(ffn_input_norm, expert_labels=expert_labels)
        else: ffn_output = self.feed_forward(ffn_input_norm)
        final_output = hidden_states + self.dropout(ffn_output)
        total_balance_loss = attn_reg_loss + moe_balance_loss; total_supervised_loss = moe_supervised_loss; total_reward = attn_reward + moe_router_reward
        return final_output, moe_usage, total_balance_loss, total_supervised_loss, total_reward

    def forward(self, x: torch.Tensor, start_pos: int, attention_mask: Optional[torch.Tensor], memory_context: Optional[torch.Tensor] = None,
                expert_labels: Optional[torch.Tensor] = None, languages: Optional[List[str]] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        use_checkpointing = self.gradient_checkpointing and self.training and _CHECKPOINT_AVAILABLE
        if use_checkpointing and self.is_on_xla is None:
            self.is_on_xla = _XLA_AVAILABLE and x.device.type == 'xla'
            if self.is_on_xla and self.layer_id == 0: logger.warning("XLA device detected: Gradient checkpointing DISABLED.")
        if use_checkpointing and self.is_on_xla: use_checkpointing = False
        if use_checkpointing:
            # Pass arguments explicitly matching _forward_impl signature
            return checkpoint.checkpoint(
                self._forward_impl, x, attention_mask, memory_context, expert_labels, languages, start_pos,
                use_reentrant=True, preserve_rng_state=True
            )
        else:
            return self._forward_impl(x, attention_mask, memory_context, expert_labels, languages, start_pos)

# --- Main Model Class ---
class Xenith(nn.Module):
    """The Xenith Transformer Model."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        if _XLA_AVAILABLE: self.device = xm.xla_device()
        elif torch.cuda.is_available(): self.device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            if torch.backends.mps.is_available() and torch.backends.mps.is_built(): self.device = torch.device("mps")
            else: self.device = torch.device("cpu"); logger.warning("MPS detected but not available/built. Using CPU.")
        else: self.device = torch.device("cpu")
        logger.info(f"--- Initializing Xenith Model on device: {self.device} ---")
        logger.info(f"--- Using ModelArgs default dtype: {self.args.model_dtype} ---")
        self.tokenizer = self._initialize_tokenizer()
        if self.tokenizer is None: raise RuntimeError("Tokenizer initialization failed.")
        if self.args.vocab_size <= 0: raise RuntimeError("ModelArgs vocab_size not updated by tokenizer.")
        logger.info(f"Tokenizer initialized. Vocab size: {self.args.vocab_size}")
        self.token_embedding = ParallelEmbedding(self.args.vocab_size, self.args.dim, self.args.model_dtype)
        self.layers = nn.ModuleList([Block(i, args) for i in range(args.n_layers)])
        norm_class = LayerNorm if args.use_layer_norm else RMSNorm
        self.norm = norm_class(args.dim, eps=args.norm_eps, dtype=self.args.model_dtype)
        self.output_layer = LoRALinear(
            args.dim, self.args.vocab_size, args.o_lora_rank, bias=False, use_lora=args.use_lora, dtype=self.args.model_dtype
        )
        if args.use_long_context_management:
            self.context_projector = LoRALinear(
                args.dim, args.dim, args.ffn_lora_rank, bias=False, use_lora=args.use_lora, dtype=self.args.model_dtype
            )
            self.summarizer = TransformerSummarizer(args)
            self.memory_bank = DynamicMemoryBank(args.dim, args.context_summary_size, args.context_memory_bank_max_chunks, self.args.model_dtype)
        else:
             self.context_projector, self.summarizer, self.memory_bank = None, None, None
             logger.info("Long context management disabled.")
        self.to(self.device)
        total_params, active_params = self.count_parameters(); logger.info(f"Parameter Count: Total={total_params:,}, Active={active_params:,}")
        logger.info(f"Gradient Checkpointing: {'Enabled (non-XLA only)' if args.use_gradient_checkpointing else 'Disabled'}")
        logger.info("--- Xenith Model Initialization Complete ---")

    def _initialize_tokenizer(self) -> Optional[AutoTokenizer]:
        if not _TRANSFORMERS_AVAILABLE: return None
        combined_path = self.args.combined_tokenizer_path; base_model_name = "xlm-roberta-large"; local_base_path = os.path.join(combined_path, "base_xlmr"); tokenizer = None
        try: tokenizer = AutoTokenizer.from_pretrained(combined_path, use_fast=True); logger.info(f"Loaded combined tokenizer from '{combined_path}'")
        except Exception as e_load:
            logger.warning(f"Could not load combined tokenizer from '{combined_path}'. Attempting creation. Error: {e_load}")
            try: tokenizer = AutoTokenizer.from_pretrained(local_base_path, use_fast=True); logger.info(f"Loaded base from '{local_base_path}'")
            except Exception:
                logger.warning(f"Local base not found at '{local_base_path}'. Downloading '{base_model_name}'...");
                try:
                    os.makedirs(local_base_path, exist_ok=True); tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True); tokenizer.save_pretrained(local_base_path); logger.info(f"Saved base to {local_base_path}")
                except Exception as e_download: logger.error(f"FAILED base download/save: {e_download}", exc_info=True); return None
            if tokenizer:
                num_added = 0; current_vocab = tokenizer.get_vocab()
                if _INDICNLP_AVAILABLE:
                    sample_hindi = "नमस्ते दुनियाँ। यह एक परीक्षण है। भाषा मॉडल ०१२३४५६७८९"; # Example
                    try:
                        hindi_tokens = set(indic_tokenize.trivial_tokenize(sample_hindi, lang='hi')); new_hindi = [t for t in hindi_tokens if t not in current_vocab]
                        if new_hindi: added = tokenizer.add_tokens(new_hindi); num_added += added; logger.info(f"Added {added} Hindi tokens.")
                    except Exception as e_indic: logger.error(f"IndicNLP failed: {e_indic}")
                else: logger.warning("IndicNLP not available, skipping Hindi addition.")
                custom_tokens = ["<HIN>", "</HIN>", "<ENG>", "</ENG>", "<MIX>", "</MIX>", "<CTX>", "</CTX>", "<SUM>", "</SUM>"]; new_custom = [t for t in custom_tokens if t not in current_vocab]
                if new_custom: added = tokenizer.add_tokens(new_custom); num_added += added; logger.info(f"Added {added} custom tokens.")
                if num_added > 0:
                    logger.info(f"Total new tokens: {num_added}")
                    try: os.makedirs(combined_path, exist_ok=True); tokenizer.save_pretrained(combined_path); logger.info(f"Saved FINAL combined tokenizer to {combined_path}")
                    except Exception as e_save: logger.error(f"FAILED save combined tokenizer: {e_save}")
            else: logger.error("Base tokenizer not loaded."); return None
        if tokenizer is None: logger.error("Tokenizer init failed."); return None
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None: tokenizer.pad_token_id=tokenizer.eos_token_id; logger.warning(f"Set pad->eos ({tokenizer.pad_token_id})")
            else: tokenizer.add_special_tokens({'pad_token': '[PAD]'}); logger.warning("Added '[PAD]' as pad token.");
            try: tokenizer.save_pretrained(combined_path); logger.info(f"Saved tokenizer after PAD update to '{combined_path}'.")
            except Exception as e_save_pad: logger.error(f"Failed save after PAD update: {e_save_pad}")
        final_vocab_size = len(tokenizer);
        if self.args.vocab_size != final_vocab_size: logger.info(f"Updating ModelArgs.vocab_size from {self.args.vocab_size} to {final_vocab_size}."); self.args.vocab_size = final_vocab_size
        else: logger.info(f"ModelArgs.vocab_size ({self.args.vocab_size}) matches tokenizer size.")
        return tokenizer

    def count_parameters(self) -> Tuple[int, int]:
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad); active_params = 0
        active_params += sum(p.numel() for p in self.token_embedding.parameters() if p.requires_grad)
        active_params += sum(p.numel() for p in self.norm.parameters() if p.requires_grad)
        active_params += sum(p.numel() for p in self.output_layer.parameters() if p.requires_grad)
        if self.context_projector: active_params += sum(p.numel() for p in self.context_projector.parameters() if p.requires_grad)
        if self.summarizer: active_params += sum(p.numel() for p in self.summarizer.parameters() if p.requires_grad)
        for l in self.layers:
            active_params += sum(p.numel() for p in l.attention.parameters() if p.requires_grad)
            active_params += sum(p.numel() for p in l.attention_norm.parameters() if p.requires_grad)
            active_params += sum(p.numel() for p in l.ffn_norm.parameters() if p.requires_grad)
            if l.language_adapter: active_params += sum(p.numel() for p in l.language_adapter.parameters() if p.requires_grad)
            if hasattr(l, 'memory_gate') and l.memory_gate: active_params += sum(p.numel() for p in l.memory_gate.parameters() if p.requires_grad)
            if l.is_moe_layer:
                m=l.feed_forward; active_params += sum(p.numel() for p in m.router.parameters() if p.requires_grad)
                if m.shared_expert_mlp: active_params += sum(p.numel() for p in m.shared_expert_mlp.parameters() if p.requires_grad)
                if m.experts: one_expert_params = sum(p.numel() for p in m.experts[0].parameters() if p.requires_grad); active_params += one_expert_params * self.args.n_activated_experts
            else: active_params += sum(p.numel() for p in l.feed_forward.parameters() if p.requires_grad)
        return total_params, int(active_params)

    def _process_long_input(self, input_ids: torch.Tensor) -> List[torch.Tensor]:
        bsz, total_len = input_ids.size(); chunk_size = self.args.context_chunk_size; chunks = list(torch.split(input_ids, chunk_size, dim=1)); last_chunk = chunks[-1]; last_chunk_len = last_chunk.size(1)
        if last_chunk_len < chunk_size:
            num_padding = chunk_size - last_chunk_len; pad_token_id = self.tokenizer.pad_token_id if self.tokenizer else 0; pad_token_id = 0 if pad_token_id is None else pad_token_id
            padding = torch.full((bsz, num_padding), pad_token_id, dtype=input_ids.dtype, device=input_ids.device); chunks[-1] = torch.cat([last_chunk, padding], dim=1)
        # logger.info(f"Processed long input ({total_len}) into {len(chunks)} chunks of size {chunk_size}.")
        return chunks

    def _summarize_chunks_and_update_memory(self, chunks: List[torch.Tensor]) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        if not self.summarizer or not self.memory_bank: logger.warning("Summarizer/Memory Bank not available."); return None, torch.tensor(0.0, device=self.device, dtype=self.args.model_dtype)
        bsz = chunks[0].size(0); total_summary_reward = torch.tensor(0.0, device=self.device, dtype=torch.float); num_chunks = len(chunks)
        # logger.info(f"Starting summarization for {num_chunks} chunks...")
        for i, chunk_ids in enumerate(chunks):
            if bsz > 1: logger.warning("Summarization uses bsz=1 for memory ops. Performance impact.")
            chunk_embeddings = self.token_embedding(chunk_ids.to(self.device))
            summary, reward = self.summarizer(chunk_embeddings[0:1])
            self.memory_bank.add(summary.squeeze(0), key=i, persistent=(i == 0))
            total_summary_reward += reward.float()
        # if rank == 0: logger.info(f"Mem bank updated. Size: {len(self.memory_bank.memory)}, Usage: {self.memory_bank.get_memory_usage_mb():.2f} MB")
        last_chunk_embeddings = self.token_embedding(chunks[-1].to(self.device)); retrieved_memory = self.memory_bank.retrieve(last_chunk_embeddings, k=5)
        if retrieved_memory is None: retrieved_memory = torch.zeros(bsz, self.args.context_summary_size, self.args.dim, device=self.device, dtype=self.args.model_dtype); logger.debug("Memory retrieve failed, using zeros.")
        elif retrieved_memory.size(0) != bsz: logger.warning(f"Retrieved memory bsz ({retrieved_memory.size(0)}) != input ({bsz}). Repeating."); retrieved_memory = retrieved_memory[0:1].expand(bsz, -1, -1)
        average_summary_reward = (total_summary_reward / num_chunks) if num_chunks > 0 else torch.tensor(0.0, device=self.device)
        # logger.info(f"Chunk summarization complete. Avg reward: {average_summary_reward.item():.4f}")
        return retrieved_memory.to(self.args.model_dtype), average_summary_reward.to(self.args.model_dtype)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, start_pos: int = 0,
                expert_labels: Optional[torch.Tensor] = None, languages: Optional[List[str]] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids = input_ids.to(self.device); bsz, total_seq_len = input_ids.shape
        if attention_mask is not None: attention_mask = attention_mask.to(self.device)
        if expert_labels is not None: expert_labels = expert_labels.to(self.device)
        active_input_ids = input_ids; active_seq_len = total_seq_len; active_attention_mask = attention_mask; active_expert_labels = expert_labels
        memory_context = None; total_summary_reward = torch.tensor(0.0, device=self.device, dtype=self.args.model_dtype)

        if self.args.use_long_context_management and total_seq_len > self.args.context_chunk_size:
            if start_pos == 0 and self.summarizer and self.memory_bank:
                 chunks = self._process_long_input(input_ids)
                 memory_context, total_summary_reward = self._summarize_chunks_and_update_memory(chunks)
                 active_input_ids = chunks[-1]; active_seq_len = active_input_ids.size(1)
                 if active_attention_mask is not None: active_attention_mask = active_attention_mask[:, -active_seq_len:]
                 if active_expert_labels is not None: active_expert_labels = active_expert_labels[:, -active_seq_len:]
                 # logger.info(f"Processing last chunk (len={active_seq_len}) with memory.")
            # elif start_pos > 0: logger.warning("Chunking disabled for generation step.") # Avoid repeated warnings
        elif start_pos > 0 and self.memory_bank is not None and len(self.memory_bank.memory) > 0:
             current_embeddings = self.token_embedding(active_input_ids)
             retrieved_memory = self.memory_bank.retrieve(current_embeddings, k=5)
             if retrieved_memory is not None: memory_context = retrieved_memory; logger.debug(f"Retrieved memory during generation (start_pos={start_pos}).")

        projected_memory_context = None
        if memory_context is not None and self.context_projector is not None: projected_memory_context = self.context_projector(memory_context)

        combined_attention_mask = None
        if active_seq_len > 1:
            causal_mask = torch.triu(torch.full((active_seq_len, active_seq_len), float('-inf'), device=self.device, dtype=self.args.model_dtype), diagonal=1).unsqueeze(0).unsqueeze(0)
            if active_attention_mask is not None:
                additive_padding_mask = torch.zeros_like(active_attention_mask, dtype=self.args.model_dtype)
                additive_padding_mask.masked_fill_(active_attention_mask == 0, float("-inf"))
                additive_padding_mask = additive_padding_mask.unsqueeze(1).unsqueeze(2)
                combined_attention_mask = causal_mask + additive_padding_mask
            else: combined_attention_mask = causal_mask

        hidden_states = self.token_embedding(active_input_ids)
        accumulated_expert_usage = None; accumulated_balance_loss = torch.tensor(0.0, device=self.device, dtype=self.args.model_dtype)
        accumulated_supervised_loss = torch.tensor(0.0, device=self.device, dtype=self.args.model_dtype); accumulated_router_reward = torch.tensor(0.0, device=self.device, dtype=self.args.model_dtype)

        for layer in self.layers:
            hidden_states, moe_usage, balance_loss, supervised_loss, router_reward = layer(
                x=hidden_states, start_pos=start_pos, attention_mask=combined_attention_mask, memory_context=projected_memory_context,
                expert_labels=active_expert_labels, languages=languages
            )
            if moe_usage is not None: accumulated_expert_usage = accumulated_expert_usage + moe_usage if accumulated_expert_usage is not None else moe_usage
            accumulated_balance_loss += balance_loss; accumulated_supervised_loss += supervised_loss; accumulated_router_reward += router_reward

        hidden_states = self.norm(hidden_states)
        logits = self.output_layer(hidden_states).float()
        if accumulated_expert_usage is not None and self.args.moe:
             num_moe_layers = sum(1 for l in self.layers if l.is_moe_layer)
             if num_moe_layers > 0: accumulated_expert_usage /= num_moe_layers
        return (logits, accumulated_expert_usage, accumulated_balance_loss, accumulated_supervised_loss, accumulated_router_reward, total_summary_reward)

    def save_pretrained(self, save_directory: str, safe_serialization: bool = True):
        """Saves the model state dict, config, and tokenizer."""
        if rank != 0: return
        os.makedirs(save_directory, exist_ok=True); logger.info(f"Saving model and tokenizer to {save_directory}")
        if self.tokenizer:
            try: self.tokenizer.save_pretrained(save_directory); logger.info(f"Tokenizer saved to {save_directory}")
            except Exception as e: logger.error(f"Failed to save tokenizer to {save_directory}: {e}", exc_info=True)
        else: logger.warning("Model has no tokenizer attribute to save.")
        # --- Config Saving Placeholder ---
        config_path = os.path.join(save_directory, "config.json"); logger.warning(f"Model config saving not fully implemented. Placeholder: {config_path}")
        # --- State Dict Saving ---
        state_dict = {k: v.cpu().clone() for k, v in self.state_dict().items() if not k.endswith("freqs_cis")} # Move to CPU and filter buffer
        weights_name = "model.safetensors" if safe_serialization and _SAFETENSORS_AVAILABLE else "pytorch_model.bin"
        save_path = os.path.join(save_directory, weights_name)
        if safe_serialization and _SAFETENSORS_AVAILABLE:
            try: save_file(state_dict, save_path); logger.info(f"Saved weights using safetensors to {save_path}")
            except Exception as e: logger.error(f"Failed save safetensors: {e}"); raise
        else:
            logger.info(f"Saving weights using torch.save to {save_path}")
            use_xm_save = _XLA_AVAILABLE and 'xla' in str(self.device) # Check if model is actually on XLA
            if use_xm_save: xm.save(state_dict, save_path)
            else: torch.save(state_dict, save_path)
        logger.info("Model saving complete.")

    @classmethod
    def from_pretrained(cls, load_directory: str, config_overrides: Optional[Dict[str, Any]] = None):
        """Loads the model state dict, config, and tokenizer."""
        logger.info(f"--- Loading Xenith Model from: {load_directory} ---")
        # --- Config Loading Placeholder ---
        logger.warning("Model config loading not fully implemented. Using default ModelArgs + overrides.")
        args = ModelArgs()
        if config_overrides:
             logger.info(f"Applying config overrides: {config_overrides}")
             for key, value in config_overrides.items():
                  if hasattr(args, key): setattr(args, key, value)
                  else: logger.warning(f"Config override key '{key}' not found.")
        # --- Set Tokenizer Path ---
        args.combined_tokenizer_path = load_directory # Expect tokenizer in the load directory
        # --- Load Weights ---
        weights_path_safe = os.path.join(load_directory, "model.safetensors")
        weights_path_bin = os.path.join(load_directory, "pytorch_model.bin")
        state_dict = None
        if os.path.exists(weights_path_safe) and _SAFETENSORS_AVAILABLE:
            logger.info(f"Loading weights from safetensors file: {weights_path_safe}")
            state_dict = load_file(weights_path_safe, device="cpu") # Load to CPU
        elif os.path.exists(weights_path_bin):
            logger.warning(f"Loading weights using torch.load from: {weights_path_bin}.")
            state_dict = torch.load(weights_path_bin, map_location="cpu")
        else: raise FileNotFoundError(f"No model weights file (model.safetensors or pytorch_model.bin) found in {load_directory}")
        if state_dict is None: raise IOError(f"Failed to load state dict from {load_directory}")
        # --- Instantiate Model (Loads Tokenizer) ---
        model = cls(args)
        # --- Load State Dict ---
        try:
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if unexpected_keys: logger.warning(f"Unexpected keys in state dict ignored: {unexpected_keys}")
            critical_missing = [k for k in missing_keys if not k.endswith("freqs_cis")]
            if critical_missing: logger.error(f"STRICT LOAD FAILED. Missing keys: {critical_missing}"); raise RuntimeError("Strict load failed")
            elif missing_keys: logger.warning(f"Missing non-critical keys (buffers): {missing_keys}")
            logger.info("Model state dict loaded successfully.")
        except Exception as e: logger.error(f"Error applying state dict: {e}", exc_info=True); raise
        logger.info(f"--- Xenith Model Loading Complete from {load_directory} ---")
        model.eval(); return model

# --- Compatibility Check Execution ---
if __name__ == "__main__":
    logger.info("--- Running Xenith Model Compatibility Check ---")
    try:
        # 1. Minimal Config for Instantiation Test
        check_args = ModelArgs(
            dim=32, n_layers=1, n_heads=2, n_kv_heads=1, # Very small dimensions
            moe=True, n_routed_experts=2, n_activated_experts=1, # Minimal MoE
            max_seq_len=64, context_chunk_size=32, context_summary_size=4, # Small context
            vocab_size=-1, # Let tokenizer handle this
            combined_tokenizer_path="./compat_test_tokenizer", # Temporary path
            model_dtype=torch.float32 # Use float32 for broader CPU compatibility check
        )
        logger.info(f"Using minimal config for check: {check_args}")

        # Clean up previous test tokenizer dir
        if os.path.exists(check_args.combined_tokenizer_path):
            try: shutil.rmtree(check_args.combined_tokenizer_path)
            except OSError as e: logger.warning(f"Could not remove temp tokenizer dir: {e}")

        # 2. Instantiate Model (includes tokenizer init)
        logger.info("Attempting model instantiation...")
        model = Xenith(check_args)
        logger.info("Model instantiation successful.")
        logger.info(f"Model on device: {model.device}")
        logger.info(f"Final vocab size: {model.args.vocab_size}")

        # 3. (Optional) Basic Forward Pass Check (CPU only if no GPU/TPU available)
        if model.device.type == 'cpu' or torch.cuda.is_available() or _XLA_AVAILABLE:
             logger.info("Attempting minimal forward pass...")
             test_input = torch.randint(0, model.args.vocab_size, (1, 16), device=model.device) # Short sequence
             model.eval() # Set eval mode
             with torch.no_grad():
                  # Use autocast appropriate for the device, disable if CPU
                  dtype_autocast = torch.bfloat16 if model.device.type != 'cpu' else torch.float32 # Prefer bfloat16 for GPU/TPU checks
                  use_autocast = model.device.type != 'cpu'
                  device_type_autocast = model.device.type if model.device.type != 'mps' else 'cpu' # MPS autocast needs CPU type

                  # Disable autocast if using float32 on XLA as it warns
                  if model.device.type == 'xla' and dtype_autocast == torch.float32:
                       use_autocast = False
                       logger.info("Disabling autocast for float32 check on XLA.")

                  with torch.autocast(device_type=device_type_autocast, dtype=dtype_autocast, enabled=use_autocast):
                       _ = model(test_input) # Just run forward, ignore output
             logger.info("Minimal forward pass successful.")
        else:
             logger.info("Skipping forward pass check (no suitable device detected).")

        # 4. Clean up temporary tokenizer
        if os.path.exists(check_args.combined_tokenizer_path):
             try: shutil.rmtree(check_args.combined_tokenizer_path)
             except OSError as e: logger.warning(f"Could not remove temp tokenizer dir: {e}")

        print("\n-----------------------------------------")
        print("✅ Xenith Model Script Compatibility Check Passed!")
        print("   Model instantiated successfully.")
        print("-----------------------------------------")

    except Exception as e:
        logger.error(f"Compatibility check failed: {e}", exc_info=True)
        print("\n-----------------------------------------")
        print("❌ Xenith Model Script Compatibility Check FAILED.")
        print("   Review the error logs above.")
        print("-----------------------------------------")
        sys.exit(1) # Exit with error code if check fails

# Add a final print statement to confirm script parsing when imported
print("Xenith model script loaded successfully (defined classes and functions).")
