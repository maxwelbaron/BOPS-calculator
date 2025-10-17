from models import MODELS
from operations import Operand,BopCounter
import os,pandas,itertools,numpy as np


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


def run_experiment(n_inputs=28, window_size=10, model_type = "Transformer", compression_type = "dense",**kwargs):
    register = BopCounter(**kwargs)
    model_hyperparams = {**PRESETS[model_type],**kwargs}
    X = Operand((window_size,n_inputs))
    return MODELS[model_type](X,register=register,type=compression_type,**model_hyperparams)[1]


def run_experiment_grid(save_path="./results.csv",skip_previous=True,bops_units="G",size_units="M",**grid):
    if not os.path.exists(save_path):
    # if True:
        df = pandas.DataFrame(columns=[*list(grid.keys()),f'size({size_units}b)',f'{bops_units}BOPs'])
        load = False
    else:
        df = pandas.read_csv(save_path)
        load = True

    params = [{k:v for k,v in zip(grid.keys(),combo)} for combo in itertools.product(*grid.values())]
    for i in range(len(params)):
        print(f'\n\n-----------------[{i+1}/{len(params)}]----------------------')
        if skip_previous and load and np.any(np.all([df[k].values == type(df[k][0])(v) for k,v in params[i].items()],axis=0)):
            print("skipping",params[i])
            continue
        register = run_experiment(bops_units=bops_units,size_units=size_units,**params[i])
        df.loc[len(df)] = {**params[i],f'size({size_units}b)':register.size(),f'{bops_units}BOPs':register.bops}
        print(df.loc[len(df)-1])
        df.to_csv(save_path,index=False)
    return df


if __name__ == "__main__":
    run_experiment_grid(
        model_type = ["Transformer","LSTM","MAMBA"],
        compression_type = ["dense","quantize_8","SB_0.25","SB_0.5"],
        window_size = [10,20,30,40,50,60],
        n_inputs = [28]
    )