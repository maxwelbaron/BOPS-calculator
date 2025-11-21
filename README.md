# Bit-Wise Storage and Computation Cost Estimator for Neural Networks

This package provides modular neural-network components built on top of an
`operations` backend that tracks computation and storage through a registry.  
Instead of performing numerical forward passes directly, each function builds a
graph of `operations` while recording costs using an
`operations.BopCounter`.

The design enables:
- **Operation-level instrumentation** (e.g., bit-operation counting)  
- **Static analysis of model structure**  
- **Backend-agnostic model definition**  
- **Composable neural blocks** for MLPs, Transformers, LSTMs, and SSMs

---

## Register mechanism

```python
operations.BopCounter()
```
The register serves three main purposes:

1. Layer Tracking: Every learnable layer (Linear, Conv1d, etc.) is inserted into the
registry under a unique ID.

2. Operation Wrapping: Forward calls are wrapped using register(...) to ensure all intermediate
operations are logged.

3. Composable Graph Construction: Any function returning (X, register) can be nested inside others without
losing operation history.

### Example Usage

```python
import operations
from blocks import MLP, transformer_encoder_block

X = operations.Operand((128, 32))   # example input
register = operations.BopCounter()

# Build an MLP
Y, register = MLP(X, n_hiddens=[64, 64], act_func="relu", register=register)

# Add a transformer block
Z, register = transformer_encoder_block(Y, model_dim=32, n_heads=4, register=register)

print(f'storage = {register.size()}{register.size_units}, computation = {register.bops}{register.bops_units}')
```

## Built-in Layers 

```python
layers.MLP(X:operations.Operand,n_outputs=None,register=None,n_hiddens=[40],act_func="relu",type="dense",**kwargs)
layers.transformer_encoder_block(X:operations.Operand,model_dim=32,depth_multiplier=2,n_heads = 4,register=None,type="dense",**kwargs)
layers.LSTM_cell(X:operations.Operand,hidden_size = 30,type="dense",register=None,**kwargs)
layers.SSM_layer(X:operations.Operand,d_state=16,type="dense",register=None,**kwargs)
layers.MAMBA_block(X:operations.Operand, register = None, d_expand=2, patch_size=4,act_func="relu", type="dense",ssm_layer_f = SSM_layer,**kwargs)
```
