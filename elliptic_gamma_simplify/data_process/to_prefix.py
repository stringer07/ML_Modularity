# Adapted from https://github.com/facebookresearch/SymbolicMathematics under CC BY-NC 4.0

import sympy as sp
from src.utils import AttrDict
from src.envs import build_env
from tqdm import tqdm
from pathlib import Path

params = AttrDict({

    # environment parameters
    'env_name': 'char_sp',
    'int_base': 10,
    'balanced': False,
    'positive': True,
    'precision': 10,
    'n_variables': 1,
    'n_coefficients': 0,
    'leaf_probs': '0.75,0,0.25,0',
    'max_len': 1024,
    'max_int': 5,
    'max_ops': 15,
    'max_ops_G': 15,
    'clean_prefix_expr': True,
    'rewrite_functions': '',
    'tasks': 'prim_fwd',
    'operators': 'add:10,sub:3,mul:10,div:5,sqrt:4,pow2:4,pow3:2,pow4:1,pow5:1,ln:4,exp:4,sin:4,cos:4,tan:4,asin:1,acos:1,atan:1,sinh:1,cosh:1,tanh:1,asinh:1,acosh:1,atanh:1',
})

data_type="random_ns_nt"
data_path=Path("./data/train")

env = build_env(params)
n = 1000000

with open (data_path/data_type/'origin_prefix.txt','w') as file_prefix:
    with open (data_path/data_type/'origin_infix.txt','r') as file_infix:
        for i in tqdm(range(n), desc="Converting to prefix notation of origin data: {}".format(data_type), unit=" lines"):
            expr_str_origin=file_infix.readline()
            expr_sp_origin=sp.S(expr_str_origin,locals=env.local_dict)
            expr_prefix_origin=env.sympy_to_prefix(expr_sp_origin)
            file_prefix.write('{}\n'.format(expr_prefix_origin))

with open (data_path/data_type/'simple_prefix.txt','w') as file_prefix:
    with open (data_path/data_type/'simple_infix.txt','r') as file_infix:
        for i in tqdm(range(n), desc="Converting to prefix notation of simple data: {}".format(data_type), unit=" lines"):
            expr_str_origin=file_infix.readline()
            expr_sp_origin=sp.S(expr_str_origin,locals=env.local_dict)
            expr_prefix_origin=env.sympy_to_prefix(expr_sp_origin)
            file_prefix.write('{}\n'.format(expr_prefix_origin))

print("droping out line >1024")
total_lines = 1000000

with open(data_path/data_type/'origin_prefix.txt', 'r') as f1, open(data_path/data_type/'simple_prefix.txt','r') as f2, \
     open(data_path/data_type/'origin_prefix_1024.txt', 'w') as f1_out, open(data_path/data_type/'simple_prefix_1024.txt', 'w') as f2_out:

    with tqdm(total=total_lines, desc="Processing", unit=" lines") as pbar:
        for line1, line2 in zip(f1, f2):

            if len(eval(line1)) <= 1022:

                f1_out.write(line1)
                f2_out.write(line2)

            pbar.update(1)

