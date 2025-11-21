import operations
import math



def MLP(X:operations.Operand,n_outputs=None,register=None,n_hiddens=[40],act_func="relu",type="dense",**kwargs):
    """
        Computes cost and symbolic forward pass of a multi-layer perceptron (MLP) using mixed-precision
        linear layers and activation functions.

        Parameters
        ----------
        X : Operand
            Input operand to the MLP. Its shape determines the input dimension.
        n_outputs : int, optional
            Output dimensionality of the final layer. Defaults to `X.shape[-1]`.
        register : BopCounter, optional
            A register object used for accumulating bit-operations and parameter
            sizes. If None, a new BopCounter is created.
        n_hiddens : list of int
            Hidden layer sizes for the MLP.
        act_func : {"relu", "tanh", "sigmoid"}
            Activation function identifier.
        type : str
            Linear layer type string (e.g., "dense", "quantize_8", "SB_0.5").
        **kwargs
            Additional keyword arguments forwarded to the Linear layer.

        Returns
        -------
        (Operand, BopCounter)
            The output operand of the MLP and the updated bop register.

        Notes
        -----
        - Each dense/quantized/sparse layer is registered and contributes its
        parameter size and bit-operation count.
        - Activation computation cost is estimated via the corresponding function
        in `operations.ACTIVATION`.
    """
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

def transformer_encoder_block(X:operations.Operand,model_dim=32,depth_multiplier=2,n_heads = 4,register=None,type="dense",**kwargs):
    """
        Compute the cost and symbolic forward pass of a Transformer encoder block.

        The block includes:
            - Multi-head self-attention (Q, K, V projections and output projection)
            - Softmax attention computation
            - Residual connections
            - Two layer-norm operations
            - Feed-forward MLP sub-block

        Parameters
        ----------
        X : Operand
            Input token sequence of shape (sequence_length, model_dim).
        model_dim : int
            Embedding dimension.
        depth_multiplier : float
            Scales the per-head projection depth as `ceil((model_dim / n_heads) * multiplier)`.
        n_heads : int
            Number of attention heads.
        register : BopCounter, optional
            Accumulator for bit-ops and parameter sizes.
        type : str
            Precision type string for Linear layers.
        **kwargs
            Additional arguments forwarded to the MLP inside the block.

        Returns
        -------
        (Operand, BopCounter)
            Output operand and updated bop register.

        Notes
        -----
        - Self-attention cost is computed via explicit matmul + softmax BOP estimates.
        - Layer normalization cost follows the formulas in `operations.layer_norm()`.
        - The feed-forward network uses the MLP inference-cost API.
    """
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
    X1,register = MLP(X,n_outputs = model_dim,type=type,register=register,**kwargs)
    X = register(operations.add(X,X1))
    register[block_id +"LN2"] = operations.LayerNorm((model_dim,))
    X = register(register[block_id +"LN2"](X))
    return X,register

def LSTM_cell(X:operations.Operand,hidden_size = 30,type="dense",register=None,**kwargs):
    """
        Compute the cost and symbolic forward pass of an LSTM cell unrolled
        through time.

        The LSTM cell includes:
            - Forget gate
            - Input gate
            - Output gate
            - Candidate cell update
            - layer norm
            - Recurrent connections

        Parameters
        ----------
        X : Operand
            Sequence input of shape (sequence_length, feature_dim).
        hidden_size : int
            Dimensionality of hidden and cell states.
        type : str
            Precision type for all Linear layers.
        register : BopCounter, optional
            Cost accumulator.

        Returns
        -------
        (Operand, BopCounter)
            Output sequence of hidden states Hs and the updated register.

        Notes
        -----
        - Explicitly simulates each time step to accumulate BOPs for gate
        operations and element-wise operations.
        - Uses the activation cost estimates from `operations.sigmoid` and
        `operations.tanh`.
        - Normalization cost is added through a per-step LayerNorm operation.
    """
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

def SSM_layer(X:operations.Operand,d_state=16,type="dense",register=None,**kwargs):
    """
        Compute the cost and symbolic forward pass of a state-space model (SSM) layer
        through time.

        Parameters
        ----------
        X : Operand
            Input sequence of shape (sequence_length, feature_dim).
        d_state : int
            Size of the latent state in the SSM.
        type : str
            Precision type for all Linear layers.
        register : BopCounter, optional
            Bit-op accumulator.

        Returns
        -------
        (Operand, BopCounter)
            Output sequence after the SSM layer and updated register.

        Notes
        -----
        - Implements a discretized linear state-space recurrence:
            H[t] = δA * H[t-1] + δB * X[t]
        - Costs include:
            • Four input projections (A, B, C, D)
            • State update matmuls
            • Elementwise multiplications and additions
            • Sigmoid gate projection and application
    """
    if register is None:
        register = operations.BopCounter()
    ssm_id = register.get_id("SSM_layer_")
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

def MAMBA_block(X:operations.Operand, register = None, d_expand=2, patch_size=4,act_func="relu", type="dense",ssm_layer_f = SSM_layer,**kwargs):
    """
        Compute the cost and symbolic forward pass of a Mamba-style block.

        The block includes:
            - Input linear expansions
            - Convolution-like projection
            - Activation + RMSNorm
            - SSM recurrence subblock
            - Gating mechanism
            - Linear output projection
            - Final RMSNorm

        Parameters
        ----------
        X : Operand
            Input sequence of shape (sequence_length, model_dim).
        register : BopCounter, optional
            Cost accumulator.
        d_expand : float
            Expansion factor before the SSM layer.
        patch_size : int
            Convolution kernel size for the Conv1d projection.
        act_func : {"relu", "tanh", "sigmoid"}
            Activation applied after convolution.
        type : str
            Precision specification for all Linear and Conv1d weights.
        ssm_layer_f : callable
            Function that computes an SSM layer and returns (Operand, register).
        **kwargs
            Additional arguments forwarded to the SSM layer.

        Returns
        -------
        (Operand, BopCounter)
            Output operand Z and updated bop register.

        Notes
        -----
        - This implementation mirrors the computational structure of Mamba/SSM
        families while serving as a symbolic engine for mixed-precision cost
        estimation.
        - Bit-ops include all linear projections, gating, elementwise ops,
        convolution cost, and SSM recurrence cost.
    """
    if register is None:
        register = operations.BopCounter()
    d_in = X.shape[-1] * d_expand
    block_id = register.get_id("MAMBA_block_")
    register[block_id + "input"] = operations.Linear(X.shape[-1],d_in,type=type)
    register[block_id + "gate"] = operations.Linear(X.shape[-1],d_in,type=type)
    register[block_id + "cnn"] = operations.Conv1d(d_in,patch_size,d_in,type=type)
    register[block_id + "output"] = operations.Linear(d_in,X.shape[-1],type=type)
    register[block_id + "norm1"] = operations.RMSNorm((d_in,)) 
    register[block_id + "norm2"] = operations.RMSNorm((X.shape[-1],))

    X_in = register(register[block_id + "input"](X))
    X_in = register(register[block_id + "cnn"](X_in))
    X_in = register(operations.ACTIVATION[act_func](X_in))
    X_in = register(register[block_id + "norm1"](X_in))
    X_in,register = ssm_layer_f(X_in,register=register,type=type,**kwargs)
    X_gate = register(register[block_id + "gate"](X))
    X_gate = register(operations.ACTIVATION[act_func](X_gate))
    Z = register(operations.multiply(X_in,X_gate))
    Z = register(register[block_id + "output"](Z))
    Z = register(operations.add(Z,X))
    Z = register(register[block_id + "norm2"](Z))
    return Z,register