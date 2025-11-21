import numpy as np,math

UNITS = {
    "G":1_000_000_000,
    "M":1_000_000,
    "K":1_000,
    "":1
}


class Operand:
    """
        Represents a tensor-like object with an associated bit-width for mixed-precision
        hardware cost estimation.

        The Operand class is a lightweight wrapper around a NumPy array used only for 
        tracking shape and bit-width. It enables symbolic propagation 
        through operations while keeping track of storage cost in bits.

        Parameters
        ----------
        shape : tuple[int]
            Shape of the operand tensor.
        bit_width : int, optional
            Number of bits used to represent each element (default: 32).

        Attributes
        ----------
        shape : tuple[int]
            Tensor dimensions.
        bit_width : int
            Precision of each element in bits.
        value : np.ndarray
            Simulated array only for shape propagation; numerical content is irrelevant.
    """
    def __init__(self,shape,bit_width=32):
        self.shape = shape
        self.bit_width = bit_width
        self.value = np.zeros(shape)
    
    def reshape(self,new_shape):
        val = self.value.reshape(new_shape)
        return Operand(val.shape,bit_width=self.bit_width)
    
    def transpose(self,dim_1,dim_2):
        val = self.value.swapaxes(dim_1,dim_2)
        return Operand(val.shape,bit_width=self.bit_width)
    
    def unsqueeze(self,dim):
        val = np.expand_dims(self.value,axis=dim)
        return Operand(val.shape,bit_width=self.bit_width)
    
    def squeeze(self,dim):
        val = self.value.squeeze(dim)
        return Operand(val.shape,bit_width=self.bit_width)
    
    def __getitem__(self,index):
        val = self.value[index]
        return Operand(val.shape,bit_width=self.bit_width)
    
    def __setitem__(self,index,value):
        self.value[index] = value.value
    
    def size(self):
        return np.prod(self.shape) * self.bit_width

def Scaler(bit_width=32):
    return Operand((1,),bit_width)

## binary primitives

def matmul(operand1, operand2, accumulator_width = 32):
    new_shape = (operand1.value @ operand2.value).shape
    n_ops = np.prod(new_shape) * operand1.shape[-1]
    return n_ops * ((operand1.bit_width * operand2.bit_width) + (2*accumulator_width)), Operand(new_shape,accumulator_width)

def add(operand1, operand2):
    new_shape = (operand1.value + operand2.value).shape
    n_ops = np.prod(new_shape)
    return n_ops * 2 * max(operand1.bit_width, operand2.bit_width), Operand(new_shape, max(operand1.bit_width, operand2.bit_width))

def multiply(operand1, operand2):
    new_shape = (operand1.value * operand2.value).shape
    n_ops = np.prod(new_shape)
    return n_ops * operand1.bit_width * operand2.bit_width, Operand(new_shape, max(operand1.bit_width, operand2.bit_width))


## unary primitives

def exp(operand,terms=5):
    n_ops = np.prod(operand.shape)
    return n_ops * ((operand.bit_width * (terms+1)) + ((operand.bit_width**2) * terms)),operand

def pow(operand,exponent):
    n_ops = np.prod(operand.shape)
    halves = math.floor(math.log2(exponent))
    return n_ops * ((operand.bit_width ** 2) * (halves + (np.mod(exponent // (2**np.arange(halves)),2) == 1).astype(int).sum() )), operand

def root(operand,exponent):
    n_ops = np.prod(operand.shape)
    return n_ops * (operand.bit_width**2) * math.log2(exponent), operand


## compund operations

def sigmoid(operand):
    bops = exp(operand)[0] + add(operand,Scaler())[0] + multiply(operand,Scaler())[0]
    return bops, operand

def tanh(operand):
    bops = (exp(operand)[0]*2) + multiply(operand,Scaler())[0] + (add(operand,Scaler())[0] * 2)
    return bops, operand

def relu(operand):
    return 0,operand

ACTIVATION = {
    "sigmoid":sigmoid,
    "tanh":tanh,
    "relu":relu
}

def rms_norm(operand,shape):
    bops = (multiply(operand,operand)[0] * 2) + add(operand,operand)[0] + pow(operand,2)[0] + root(Operand(shape,operand.bit_width),2)[0] + multiply(Operand(shape,operand.bit_width),Scaler())[0]
    return bops, operand

def layer_norm(operand,shape):
    bops = (add(operand,operand)[0]*5) +  (multiply(operand,operand)[0] * 4) + pow(operand,2)[0] + root(Operand(shape,operand.bit_width),2)[0]
    return bops, operand

def softmax(operand):
    bops = exp(operand)[0] + add(operand,operand)[0] + multiply(operand,operand)[0]
    return bops, operand


def quantize_linear(weight,activation,q_bits=8,accumulator_width=32):
    bops1,_ = multiply(activation,Scaler())
    bops,output = matmul(Operand(activation.shape,q_bits),weight,accumulator_width=accumulator_width) 
    bops = bops + bops1 + multiply(output,Scaler())[0]
    return bops,output

def sb_linear(weight,activation,prune_rate = 0.5,accumulator_width=32):
    bops,output = matmul(activation,weight,accumulator_width=accumulator_width)
    bops *= (1-prune_rate)
    bops +=  multiply(output,Scaler())[0]
    return bops,output

class Linear(Operand):
    """
        Operand used to estimate BOPs and storage cost of a parameterized linear transformation 
        with support for multiple precision and sparsity schemes.

        This is a subclass of Operand, but represents a *weight matrix* rather than 
        a generic activation tensor. Different types include:
            - dense (standard FP32 or configured width)
            - quantize (uniform quantization, optional 2-bit/4-bit/etc.)
            - SB (structured block sparsity with prune rate)

        Parameters
        ----------
        *shape : int
            Dimensions of the weight matrix (in_features, out_features).
        type : str, optional
            Encoding of the linear layer type:
                "dense" → normal FP32
                "quantize" or "quantize_<wbits>_<abits>"
                "SB_<prune_rate>"
        bias_width : int, optional
            Bit-width used for bias accumulation (default: 32).

        Attributes
        ----------
        type : list[str]
            Parsed type specification fields.
        scaling_param_size : int
            Additional bits required for quantization scales or sparsity encoding.
        operation : callable
            Function computing bit-ops for the chosen linear type.
        bias_width : int
            Bit-width for accumulator/bias.

        Methods
        -------
        __call__(X)
            Returns (bops,operand) for symbolic propagation.
        size()
            Returns total parameter storage in bits
    """
    def __init__(self,*shape,type="dense",bias_width = 32):
        self.type = type.split("_")
        self.scaling_param_size = 0
        if self.type[0] == "quantize":
            super().__init__(shape,8 if len(self.type)==1 else int(self.type[1]))
            self.operation = lambda ws,X: quantize_linear(ws,X,q_bits = self.bit_width if len(self.type) < 3 else int(self.type[2]),accumulator_width=bias_width)
            self.scaling_param_size = 2 * bias_width
        elif self.type[0] == "SB":
            super().__init__(shape,1)
            self.operation = lambda ws,X: sb_linear(ws,X,prune_rate  = float(self.type[1]),accumulator_width=bias_width)
            self.scaling_param_size = bias_width + np.prod(shape)
        elif self.type[0] == "dense":
            super().__init__(shape,32)
            self.operation = lambda ws,X: matmul(X,ws,accumulator_width=bias_width)
        else:
            raise Exception(f"Unkown type: {type}")
        self.bias_width = bias_width
        
    def size(self):
        return super().size() + (self.shape[-1] * self.bias_width) + self.scaling_param_size
    def __call__(self,X):
        return self.operation(self,X)
    
class LayerNorm(Operand):
    """
        Symbolic LayerNorm operator that computes the BOPs and storage cost 
        associated with layer normalization.

        Methods
        -------
        __call__(X)
            Returns (bops, Operand) for symbolic propagation.
        size()
            Returns total parameter storage in bits
    """
    def __call__(self,X):
        return layer_norm(X,(np.prod(self.shape),))
    
    def size(self):
        return super().size() * 2

class RMSNorm(Operand):
    """
        Symbolic RMSNorm operator that computes the BOPs and storage cost 
        associated with root-mean-squared normalization.

        Methods
        -------
        __call__(X)
            Returns (bops, Operand) for symbolic propagation.
    """
    def __call__(self,X):
        return rms_norm(X,(np.prod(self.shape),))


class Conv1d(Linear):
    """
    Symbolic 1D convolution operator that computes the BOPs and storage cost for a 1D convolution 
    with stride=1, dialation=1, and padding='same'

    Parameters
    ----------
    shape : tuple of ints
        Dimensions of the weight matrix (in_channels, kernel_size, out_channels).
    type : str, optional
        Encoding of the linear layer type:
            "dense" → normal FP32
            "quantize" or "quantize_<wbits>_<abits>"
            "SB_<prune_rate>"
    bias_width : int, optional
        Bit-width used for bias accumulation (default: 32).
    Methods
    -------
    __call__(X)
        Computes bit-ops using a reshaped weight matrix and a flattened input.
    """
    def __call__(self,X):
        X = Operand((X.shape[0],self.shape[0]*self.shape[1]),self.bit_width)
        return self.operation(self.reshape((self.shape[0] * self.shape[1],self.shape[2])),X)
    
class BopCounter:
    """
        Global counter and registry for tracking bit-operations (BOPs) and parameter
        storage across an entire neural network architecture.

        BopCounter abstracts accumulation, naming, formatting, and unit conversion
        for both compute and storage metrics.

        Parameters
        ----------
        bops_units : str, optional
            Unit prefix for BOP reporting ('G', 'M', 'K', or ''). Default is 'G'.
        size_units : str, optional
            Unit prefix for parameter size reporting ('M', 'K', or ''). Default is 'M'.

        Attributes
        ----------
        bops : float
            Accumulated bit-operations normalized by the chosen base unit.
        parameters : dict[str, Operand]
            Mapping of parameter names to their Operand representations.
        blocks : dict[str, int]
            Counter used to assign unique IDs to repeated blocks (e.g., layers).
        bops_units : str
            Human-readable suffix for BOPs.
        size_units : str
            Human-readable suffix for parameter storage.
        bops_base : int
            Unit scaling factor (e.g., 1e9 for 'G').
        size_base : int
            Unit scaling factor for storage reporting.

        Methods
        -------
        get_id(block_name)
            Assigns and returns a unique layer/block identifier.
        __call__(op)
            Accumulates BOPs from a tuple returned by an operation function (bops, operand) and returns operand.
        __setitem__(name, parameter)
            Registers a parameter Operand under a given name.
        __getitem__(name)
            Retrieves a registered parameter and prints its shape.
        size()
            Returns total parameter storage in chosen units.
    """
    def __init__(self,bops_units="G",size_units="M",**kwargs):
        print("creating new register")
        self.bops = 0
        self.parameters = {}
        self.blocks = {}
        self.bops_units = f'{bops_units}BOPs'
        self.size_units = f'{size_units}b'
        self.bops_base = UNITS[bops_units]
        self.size_base = UNITS[size_units]
    
    def get_id(self,block_name):
        self.blocks[block_name] = self.blocks.get(block_name,0) + 1
        return f'{block_name}{self.blocks[block_name]}'

    def __call__(self,op):
        self.bops += (op[0] /self.bops_base)
        return op[1]
    
    def __setitem__(self,name,parameter):
        if parameter in self.parameters:
            print(f"Warning: {name} already exists")
        self.parameters[name] = parameter
    
    def __repr__(self):
        string = f"BOPs counted: {self.bops/self.bops_base}{self.bops_units}\n"
        for k,v in self.parameters.items():
            string += k + "\n"
            string += "\t" + str(v) + "\n"
            string += "\t size:" + str(v.size()/self.size_base) + " " + self.size_units + "\n"
        return string
    
    def __str__(self):
        return self.__repr__()

    
    def __getitem__(self,name):
        print(name,self.parameters[name].shape)
        return self.parameters[name]
    
    def size(self):
        return np.sum([v.size()/self.size_base for v in self.parameters.values()])