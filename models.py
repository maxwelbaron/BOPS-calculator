import numpy as np,math
import operations


def mlp(X,n_outputs=None,register=None,n_hiddens=[40],act_func="relu",type="dense",**kwargs):
    if register is None:
        register = operations.BopCounter()
    if n_outputs is None:
        n_outputs = X.shape[-1]
    n_inputs  = X.shape[-1]
    for i,nh in enumerate(n_hiddens + [n_outputs]):
        layer_id = register.get_id("MLP_layer")
        register[layer_id] = operations.Linear(n_inputs,nh,type=type)
        X = register(register[layer_id](X))
        if i == len(n_hiddens):
            break
        X = register(operations.ACTIVATION[act_func](X))
        n_inputs = nh
    return X,register





def transformer_encoder_block(X,model_dim=32,depth_multiplier=2,n_heads = 4,register=None,type="dense",**kwargs):
    depth = math.ceil((model_dim /n_heads) *depth_multiplier)
    if register is None:
        register = operations.BopCounter()
    block_id = register.get_id("Transformer_block_")
    register[block_id + "WQ"] = operations.Linear(model_dim,depth * n_heads,type=type)
    register[block_id +"WK"] = operations.Linear(model_dim,depth * n_heads,type=type)
    register[block_id +"WV"] = operations.Linear(model_dim,depth * n_heads,type=type)
    register[block_id +"WC"] = operations.Linear(depth * n_heads,model_dim,type=type)
    Q = register(register[block_id +"WQ"](X)).reshape((X.shape[0],n_heads,-1)).transpose(0,1)
    K = register(register[block_id +"WK"](X)).reshape((X.shape[0],n_heads,-1)).transpose(0,1)
    V = register(register[block_id +"WV"](X)).reshape((X.shape[0],n_heads,-1)).transpose(0,1)
    QK = register(operations.matmul(Q,K.transpose(1,2)))
    register(operations.multiply(QK,operations.Scaler()))
    register(operations.softmax(QK))
    QKV = register(operations.matmul(QK,V)).transpose(0,1).reshape((X.shape[0],-1))
    X1 = register(register[block_id +"WC"](QKV))
    X = register(operations.add(X,X1))
    register[block_id +"LN1"] = operations.LayerNorm((model_dim,))
    X = register(register[block_id +"LN1"](X))
    X1,register = mlp(X,n_outputs = model_dim,type=type,register=register,**kwargs)
    X = register(operations.add(X,X1))
    register[block_id +"LN2"] = operations.LayerNorm((model_dim,))
    X = register(register[block_id +"LN2"](X))
    return X,register

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
        X,register = transformer_encoder_block(X,model_dim=model_dim,register=register,type=type,n_heads=n_heads,**kwargs)
    
    register["output"] = operations.Linear(model_dim,n_outputs,type="dense")
    Y = register(register["output"](operations.Operand((1,X.shape[-1]))))
    return Y,register




def LSTM_cell(X,hidden_size = 30,type="dense",register=None,**kwargs):
    if register is None:
        register = operations.BopCounter()
    cell_id = register.get_id("LSTM_cell_")
    register[cell_id + "forget"] = operations.Linear(X.shape[-1] + hidden_size,hidden_size,type=type)
    register[cell_id + "input"] = operations.Linear(X.shape[-1] + hidden_size,hidden_size,type=type)
    register[cell_id + "output"] = operations.Linear(X.shape[-1] + hidden_size,hidden_size,type=type)
    register[cell_id + "cell"] = operations.Linear(X.shape[-1] + hidden_size,hidden_size,type=type)
    register[cell_id + "norm"] = operations.LayerNorm((hidden_size,))

    H = operations.Operand((hidden_size,))
    C = operations.Operand((hidden_size,))
    Hs = operations.Operand((X.shape[0],hidden_size))
    for i in range(X.shape[0]):
        input = operations.Operand((X.shape[-1]+hidden_size,))
        fg = register(register[cell_id + "forget"](input))
        fg = register(operations.sigmoid(fg))
        ig = register(register[cell_id + "input"](input))
        ig = register(operations.sigmoid(ig))
        og = register(register[cell_id + "output"](input))
        og = register(operations.sigmoid(og))
        c_tilda = register(register[cell_id + "cell"](input))
        c_tilda = register(operations.tanh(c_tilda))
        fc = register(operations.multiply(fg,C))
        ic = register(operations.multiply(ig,c_tilda))
        C = register(operations.add(fc,ic))
        H = register(operations.multiply(og,register(operations.tanh(C))))
        H = register(register[cell_id + "norm"](H))
        Hs[i] = H
    return Hs,register


def LSTM(X, n_outputs = None, register = None, lstm_cells=1, **kwargs):
    if register is None:
        register = operations.BopCounter()
    if n_outputs is None:
        n_outputs = X.shape[-1]
    for cell in range(lstm_cells):
        X,register = LSTM_cell(X, register=register, **kwargs)
    return mlp(X[-1],n_outputs, register = register, **kwargs)



def SSM(X,d_state=16,type="dense",register=None,**kwargs):
    if register is None:
        register = operations.BopCounter()
    ssm_id = register.get_id("SSM_")
    d_in = X.shape[-1]
    register[ssm_id + "WA"] = operations.Linear(d_in,d_state,type=type)
    register[ssm_id + "WB"] = operations.Linear(d_in,d_state,type=type)
    register[ssm_id + "WC"] = operations.Linear(d_in,d_state,type=type)
    register[ssm_id + "WD"] = operations.Linear(d_in,d_in,type=type)
    register[ssm_id + "Wdelta"] = operations.Linear(d_in,d_in,type=type)
    
    A = register(register[ssm_id + "WA"](X)).unsqueeze(-2)
    B = register(register[ssm_id + "WB"](X)).unsqueeze(-2)
    C = register(register[ssm_id + "WC"](X)).unsqueeze(-1)
    D = register(register[ssm_id + "WD"](X))
    delta = register(register[ssm_id + "Wdelta"](X))

    delta = register(operations.sigmoid(delta)).unsqueeze(-1)
    deltaBX = register(operations.multiply(
        register(operations.multiply(B,delta)),
        X.unsqueeze(dim=-1)
    ))
    deltaA = register(operations.multiply(delta,A))

    H = operations.Operand((d_in,d_state))
    Hs =  operations.Operand((X.shape[0],d_in,d_state))
    for i in range(X.shape[0]):
        H = register(operations.multiply(deltaA[i],H))
        H = register(operations.add(H,deltaBX[i]))
        Hs[i] = H
    Y = register(operations.matmul(Hs,C)).squeeze(-1)
    Y = register(operations.add(Y,D))
    return Y, register

def MAMBA_block(X, register = None, d_expand=2, patch_size=4,act_func="relu", type="dense",**kwargs):
    if register is None:
        register = operations.BopCounter()
    d_in = X.shape[-1] * d_expand
    block_id = register.get_id("MAMBABlock_")
    register[block_id + "input"] = operations.Linear(X.shape[-1],d_in,type=type)
    register[block_id + "gate"] = operations.Linear(X.shape[-1],d_in,type=type)
    register[block_id + "cnn"] = operations.Conv1d(d_in,d_in,patch_size)
    register[block_id + "output"] = operations.Linear(d_in,X.shape[-1],type=type)
    register[block_id + "norm1"] = operations.RMSNorm((d_in,)) 
    register[block_id + "norm2"] = operations.RMSNorm((X.shape[-1],))

    X_in = register(register[block_id + "input"](X))
    X_in = register(register[block_id + "cnn"](X_in))
    X_in = register(operations.ACTIVATION[act_func](X_in))
    X_in = register(register[block_id + "norm1"](X_in))
    X_in,register = SSM(X_in,register=register,type=type,**kwargs)
    X_gate = register(register[block_id + "gate"](X))
    X_gate = register(operations.ACTIVATION[act_func](X_gate))
    Z = register(operations.multiply(X_in,X_gate))
    Z = register(register[block_id + "output"](Z))
    Z = register(operations.add(Z,X))
    Z = register(register[block_id + "norm2"](Z))
    return Z,register

def MAMBA(X, n_outputs=None, model_dim=40, register=None, type="dense", n_blocks=2, **kwargs):
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
        X,register = MAMBA_block(X,register=register,type=type,**kwargs)
    Y = register(register["output"](X[-1]))
    return Y, register


MODELS = {
    "MAMBA":MAMBA,
    "Transformer":transformer,
    "LSTM":LSTM
}