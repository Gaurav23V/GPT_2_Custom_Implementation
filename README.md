# GPT-2 Custom Implementation

A from-scratch implementation of the GPT-2 (124M parameters) transformer architecture using PyTorch, demonstrating deep understanding of transformer models and language model internals.

## Purpose

This project implements the GPT-2 architecture entirely from scratch, allowing for a complete understanding of how transformer-based language models work. The implementation includes all core components: multi-head attention, feed-forward networks, layer normalization, and the full GPT-2 architecture.

## Tech Stack

- **Language:** Python 3.x
- **Deep Learning Framework:** PyTorch
- **Tokenization:** tiktoken (GPT-2 tokenizer)
- **Pretrained Weights:** Hugging Face Transformers (for weight loading comparison)

## Implementation Details

### Architecture Components

```
gpt_2_custom.py / train_gpt2.py
├── CasualSelfAttention    # Multi-head causal self-attention
│   ├── Key, Query, Value projections
│   ├── Scaled dot-product attention
│   └── Output projection
├── MLP                     # Feed-forward network
│   ├── Linear expansion (4x embedding)
│   ├── GELU activation
│   └── Linear projection back
├── Block                   # Transformer block
│   ├── LayerNorm 1
│   ├── CasualSelfAttention
│   ├── LayerNorm 2
│   └── MLP
└── GPT                      # Main model class
    ├── Token embeddings (wte)
    ├── Position embeddings (wpe)
    ├── Transformer blocks (n_layer times)
    ├── Final LayerNorm
    └── Language model head
```

### GPT Configurations

| Model | Parameters | n_layer | n_head | n_embd |
|-------|------------|---------|--------|--------|
| GPT-2 Small | 124M | 12 | 12 | 768 |
| GPT-2 Medium | 350M | 24 | 16 | 1024 |
| GPT-2 Large | 774M | 36 | 20 | 1280 |
| GPT-2 XL | 1558M | 48 | 25 | 1600 |

### Key Implementation Features

1. **Causal Self-Attention:**
   - Implements masked self-attention to prevent attending to future tokens
   - Uses PyTorch's `scaled_dot_product_attention` for efficiency
   - Alternative manual implementation included for educational purposes

2. **Weight Sharing:**
   - Token embedding weights shared with language model head
   - Reduces parameters and improves training stability

3. **Weight Initialization:**
   - Proper initialization scaled by number of layers
   - NaN-GPT scale initialization for residual connections

4. **Pretrained Weight Loading:**
   - `from_pretrained()` method loads weights from Hugging Face
   - Handles Conv1D to Linear weight transposition
   - Validates shape matching between custom and pretrained models

5. **Data Loading:**
   - Custom `DataLoaderLite` class for efficient batch creation
   - Uses tiktoken for GPT-2 BPE tokenization

## Installation

```bash
pip install torch tiktoken transformers
```

## Usage

### Running Pretrained Model

```python
from gpt_2_custom import GPT

# Load pretrained GPT-2 weights
model = GPT.from_pretrained('gpt2')
model.eval()

# Generate text
import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
# ... generation code
```

### Training from Scratch

```python
from gpt_2_custom import GPT, GPTConfig, DataLoaderLite

model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
model = torch.compile(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

train_loader = DataLoaderLite(B=4, T=1024)

for epoch in range(num_epochs):
    x, y = train_loader.next_batch()
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
```

## Files

- **`gpt_2_custom.py`** - Full implementation with training loop and pretrained weight loading
- **`train_gpt2.py`** - Clean architecture implementation for text generation
- **`play.ipynb`** - Jupyter notebook for experimentation
- **`input.txt`** - Training data (text corpus)

## Notes

- The implementation achieves comparable performance to the original GPT-2
- Uses PyTorch compilation (`torch.compile`) for optimization
- Supports CUDA, MPS (Apple Silicon), and CPU backends
- Training uses gradient clipping for stability