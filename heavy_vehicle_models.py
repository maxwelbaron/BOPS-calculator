import numpy as np,math
import operations
import layers

def transformer(X,n_outputs=None,n_heads = 4,model_dim=32,n_blocks=3,type="dense",register=None,**kwargs):
    model_dim = model_dim + (model_dim % n_heads)
    if n_outputs is None:
        n_outputs = X.shape[-1]
    
    if register is None:
        register = operations.BopCounter()
    register["embedding"] = operations.Linear(X.shape[-1],model_dim,type="dense" if "SB" in type else type)
    register["PE"] = operations.Operand((X.shape[0],model_dim))

    X = register(register["embedding"](X))
    X = register(operations.add(X,register["PE"]))

    for _ in range(n_blocks):
        X,register = layers.transformer_encoder_block(X,model_dim=model_dim,register=register,type=type,n_heads=n_heads,**kwargs)
    
    register["output"] = operations.Linear(model_dim,n_outputs,type="dense")
    Y = register(register["output"](operations.Operand((1,X.shape[-1]))))
    return Y,register


def LSTM(X, n_outputs = None, register = None, lstm_cells=1, **kwargs):
    if register is None:
        register = operations.BopCounter()
    if n_outputs is None:
        n_outputs = X.shape[-1]
    for cell in range(lstm_cells):
        X,register = layers.LSTM_cell(X, register=register, **kwargs)
    return layers.MLP(X[-1],n_outputs, register = register, **kwargs)



def SSM(X, n_outputs=None, model_dim=40, register=None, type="dense", n_blocks=2, **kwargs):
    if register is None:
        register = operations.BopCounter()
    if n_outputs is None:
        n_outputs = X.shape[-1]
    register["embedding"] = operations.Linear(X.shape[-1],model_dim,type="dense" if "quantize" in type else type)
    register["norm"] = operations.RMSNorm((model_dim,))
    register["output"] = operations.Linear(model_dim,n_outputs,type="dense")

    X = register(register["embedding"](X))
    X = register(register["norm"](X))
    for _ in range(n_blocks):
        X,register = layers.MAMBA_block(X,register=register,type=type,ssm_layer_f=layers.SSM_layer,**kwargs)
    Y = register(register["output"](X[-1]))
    return Y, register


MODELS = {
    "SSM":SSM,
    "Transformer":transformer,
    "LSTM":LSTM
}


PRESETS = {
    "Transformer":{
        "act_func":"relu",
        "n_blocks":3,
        "model_dim":32,
        "n_hiddens":[128],
        "depth_multiplier":2,
        "n_heads":24
    },
    "LSTM":{
        "act_func":"tanh",
        "n_hiddens":[40],
        "hidden_size":160,
    },
    "MAMBA":{
        "patch_size":4,
        "d_expand":2,
        "model_dim":20,
        "n_blocks":1,
        "act_func":"relu",
        "d_state":10,
    }
}
