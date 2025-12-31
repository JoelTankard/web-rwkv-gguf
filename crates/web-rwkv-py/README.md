# web-rwkv-py

Python bindings for the web-rwkv RWKV inference engine.

## Installation

### From source (requires Rust and maturin)

```bash
cd crates/web-rwkv-py
pip install maturin
maturin develop --release
```

### Build wheel

```bash
maturin build --release
pip install target/wheels/web_rwkv_py-*.whl
```

## Usage

```python
import web_rwkv_py as wrp

# Load model
model = wrp.Model("path/to/model.gguf")

# Load tokenizer
tokenizer = wrp.Tokenizer("path/to/tokenizer.json")

# Encode text
tokens = tokenizer.encode("Hello, world!")

# Run inference
logits = model.run(tokens)

# Get embeddings (skips head projection - faster)
embeddings = model.embed(tokens)

# Get state as numpy array
state = model.back_state_numpy()

# Save and restore state
saved_state = model.back_state()
model.clear_state()
model.load_state(saved_state)
```

## API

### Model

-   `Model(path, quant=0, quant_nf4=0, quant_sf4=0)` - Load a model
-   `model.info()` - Get model info
-   `model.run(tokens, token_chunk_size=128, option=None)` - Run inference
-   `model.embed(tokens, token_chunk_size=128, last_only=True)` - Get embeddings (faster, skips head)
-   `model.back_state(device=StateDevice.Cpu)` - Get current state
-   `model.back_state_numpy(device=StateDevice.Cpu)` - Get state as numpy array
-   `model.load_state(state)` - Load a saved state
-   `model.clear_state()` - Reset state to initial

### Tokenizer

-   `Tokenizer(path)` - Load tokenizer from JSON file
-   `tokenizer.encode(text)` - Encode text to tokens
-   `tokenizer.decode(tokens)` - Decode tokens to bytes

### PyRnnOption

-   `PyRnnOption.Last` - Output logits for last token only
-   `PyRnnOption.Full` - Output logits for all tokens
-   `PyRnnOption.EmbedLast` - Output embedding for last token (skips head projection)
-   `PyRnnOption.EmbedFull` - Output embeddings for all tokens (skips head projection)
