import numpy as np,math

UNITS = {
    "G":1_000_000_000,
    "M":1_000_000,
    "K":1_000,
    "":1
}


class Operand:
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
    # return n_ops * ((operand.bit_width ** 7) + (operand.bit_width * 12)), operand

def pow(operand,exponent):
    n_ops = np.prod(operand.shape)
    halves = math.floor(math.log2(exponent))
    return n_ops * ((operand.bit_width ** 2) * (halves + (np.mod(exponent // (2**np.arange(halves)),2) == 1).astype(int).sum() )), operand

def root(operand,exponent):
    return pow(operand, exponent)


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
    def __init__(self,*shape,type="dense",bias_width = 32):
        self.type = type.split("_")
        if self.type[0] == "quantize":
            super().__init__(shape,8 if len(self.type)==1 else int(self.type[1]))
            self.operation = lambda X: quantize_linear(self,X,q_bits = self.bit_width if len(self.type) < 3 else int(self.type[2]),accumulator_width=bias_width)
        elif self.type[0] == "SB":
            super().__init__(shape,1)
            self.operation = lambda X: sb_linear(self,X,prune_rate  = float(self.type[1]),accumulator_width=bias_width)
        elif self.type[0] == "dense":
            super().__init__(shape,32)
            self.operation = lambda X: matmul(X,self,accumulator_width=bias_width)
        else:
            raise Exception(f"Unkown type: {type}")
        self.bias_width = bias_width
        
    def size(self):
        return super().size() + (self.shape[-1] * self.bias_width)
    def __call__(self,X):
        return self.operation(X)
    
class LayerNorm(Operand):
    def __call__(self,X):
        return layer_norm(X,(np.prod(self.shape),))
    
    def size(self):
        return super().size() * 2

class RMSNorm(Operand):
    def __call__(self,X):
        return rms_norm(X,(np.prod(self.shape),))
    

class Conv1d(Operand):
    def __init__(self,n_inputs,n_outputs,patch_size,bit_width=32,bias_width=32):
        super().__init__((patch_size*n_inputs,n_outputs),bit_width=bit_width)
        self.bias_width = bias_width
        self.bias_dim = n_outputs

    def __call__(self,X):
        X = Operand((X.shape[0],self.shape[0]),self.bit_width)
        return matmul(X,self,accumulator_width=self.bit_width)
    
    def size(self):
        return super().size() + (self.bias_dim * self.bias_width)
    
class BopCounter:
    def __init__(self,bops_units="G",size_units="M",**kwargs):
        print("creating new register")
        self.bops = 0
        self.parameters = {}
        self.blocks = {}
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
    
    def __getitem__(self,name):
        print(name,self.parameters[name].shape)
        return self.parameters[name]
    
    def size(self):
        return np.sum([v.size()/self.size_base for v in self.parameters.values()])