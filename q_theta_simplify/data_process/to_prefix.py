# Adapted from https://github.com/facebookresearch/SymbolicMathematics under CC BY-NC 4.0

import sympy as sp
from src.utils import AttrDict
from src.envs import build_env
import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial


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
    'max_len': 512,
    'max_int': 5,
    'max_ops': 15,
    'max_ops_G': 15,
    'clean_prefix_expr': True,
    'rewrite_functions': '',
    'tasks': 'prim_fwd',
    'operators': 'add:10,sub:3,mul:10,div:5,sqrt:4,pow2:4,pow3:2,pow4:1,pow5:1,ln:4,exp:4,sin:4,cos:4,tan:4,asin:1,acos:1,atan:1,sinh:1,cosh:1,tanh:1,asinh:1,acosh:1,atanh:1',
})


_env = None

def init_worker():

    global _env
    _env = build_env(params)

def convert_single_expr(expr_str):
    global _env
    expr_str = expr_str.strip()
    if not expr_str:
        return ''  
    try:
        expr_sp = sp.S(expr_str, locals=_env.local_dict)
        expr_prefix = _env.sympy_to_prefix(expr_sp)
        return str(expr_prefix)
    except Exception as e:
        print(f"Warning: Failed to convert '{expr_str[:50]}...': {e}")
        return ''


def convert_file_parallel(input_path, output_path, desc, num_workers=None, chunksize=100):

    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)  
    
    print(f"Reading {input_path}...")
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    print(f"Total lines: {total_lines}, using {num_workers} workers, chunksize={chunksize}")
    
    with Pool(processes=num_workers, initializer=init_worker) as pool:
        results = pool.imap(convert_single_expr, lines, chunksize=chunksize)
        
        with open(output_path, 'w') as f_out:
            for result in tqdm.tqdm(results, total=total_lines, desc=desc, unit='expr'):
                f_out.write(f'{result}\n')
    
    print(f"Done! Output saved to {output_path}")


if __name__ == '__main__':
    NUM_WORKERS = 5  
    CHUNKSIZE = 1000  
    
    convert_file_parallel(
        input_path='./data/train/random_ns_nt/infix_origin.txt',
        output_path='./data/train/random_ns_nt/prefix_origin.txt',
        desc='converting origin data',
        num_workers=NUM_WORKERS,
        chunksize=CHUNKSIZE
    )
    
    convert_file_parallel(
        input_path='./data/train/random_ns_nt/infix_simple.txt',
        output_path='./data/train/random_ns_nt/prefix_simple.txt',
        desc='converting simple data',
        num_workers=NUM_WORKERS,
        chunksize=CHUNKSIZE
    )




'''expr_str_simple='q_theta(-z/(4*t - 3), (3*t - 2)/(4*t - 3))/(q_theta(-z/(t - 7), -1/(t - 7))*q_theta(-z/(t - 1), (5*t - 4)/(t - 1)))'
print('origin expression:\n{}\n\n'.format(expr_str_simple))

expr_sp_simple=sp.S(expr_str_simple,locals=env.local_dict)
expr_prefix_simple=env.sympy_to_prefix(expr_sp_simple)
print('prefix:\n{}\n\n'.format(expr_prefix_simple))
print(type(expr_prefix_simple))
print(len(expr_prefix_simple))

#expr_infix=env.prefix_to_infix(['pow', 'q_theta', 'mul', 'INT-', '1', 'mul', 'z', 'pow', 'add', 'INT-', '7', 't', 'INT-', '2', 'mul', 'INT+', '1', 'INT-', '3'])
expr_infix=env.prefix_to_infix(['mul', 'pow', 'q_theta', 'mul', 'INT-', '1', 'mul', 'z', 'pow', 'add', 'INT+', '7', 'mul', 'INT+', '9', 't', 'INT-', '1', 'mul', 'pow', 'add', 'INT+', '7', 'mul', 'INT+', '9', 't', 'INT-', '1', 'add', 'INT-', '3', 'mul', 'INT-', '4', 't', 'INT-', '1', 'mul', 'q_theta', 'mul', 'z', 'pow', 'add', 'INT+', '4', 'mul', 'INT+', '5', 't', 'INT-', '1', 'mul', 'pow', 'add', 'INT+', '4', 'mul', 'INT+', '5', 't', 'INT-', '1', 'add', 'INT+', '1', 't', 'q_theta', 'mul', 'INT-', '1', 'mul', 'z', 'pow', 'add', 'INT+', '1', 'mul', 'INT+', '3', 't', 'INT-', '1', 'mul', 't', 'pow', 'add', 'INT+', '1', 'mul', 'INT+', '3', 't', 'INT-', '1'])
#expr_infix=env.prefix_to_infix(['mul', 'pow', 'q_theta', 'mul', 'INT-', '1', 'mul', 'z', 'pow', 'add', 'INT-', '1', 't', 'INT-', '1', 'mul', 'pow', 'add', 'INT-', '1', 't', 'INT-', '1', 'add', 'INT-', '4', 'mul', 'INT+', '5', 't', 'INT-', '1', 'mul', 'pow', 'q_theta', 'mul', 'INT-', '1', 'mul', 'z', 'pow', 'add', 'INT-', '7', 't', 'INT-', '1', 'mul', 'INT-', '1', 'pow', 'add', 'INT-', '7', 't', 'INT-', '1', 'INT-', '1', 'q_theta', 'mul', 'INT-', '1', 'mul', 'z', 'pow', 'add', 'INT-', '3', 'mul', 'INT+', '4', 't', 'INT-', '1', 'mul', 'pow', 'add', 'INT-', '3', 'mul', 'INT+', '4', 't', 'INT-', '1', 'add', 'INT-', '2', 'mul', 'INT+', '3', 't'])
#expr_infix=env.prefix_to_infix(expr_prefix_simple)
#print('infix:{}'.format(expr_infix))

expr_sympy=env.infix_to_sympy(expr_infix)
print('after switch:\n{}\n\n'.format(expr_sympy))


print('are they equal? : ',end='')
if sp.S(str(expr_str_simple))==sp.S(str(expr_sympy)):
    print('YES')
    print(sp.S(str(expr_str_simple))-sp.S(str(expr_sympy)))
else:
    print('NO')
    print(sp.S(expr_str_simple))
    print(sp.S(expr_sympy))
    print(sp.sympify(sp.S(expr_str_simple)-sp.S(expr_sympy)))'''
