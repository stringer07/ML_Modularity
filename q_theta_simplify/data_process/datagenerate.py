import sympy as sp
from sympy import Poly
import random
from sympy.core.function import Function
import tqdm
from multiprocessing import Pool
import os


q_theta=Function('q_theta')
z=sp.Symbol('z')
t=sp.Symbol('t')

VALID_COEFFS = []
for _k in range(-9, 10):
    for _l in range(-9, 10):
        for _m in range(-9, 10):
            for _n in range(-9, 10):
                if (_k * _n - _m * _l) == 1:
                    VALID_COEFFS.append((_k, _l, _m, _n))

def generate_random_args(variable1, variable2, simple_form='cancel'):
    k, l, m, n = random.choice(VALID_COEFFS)
    coeffs_num_kl = [k, l]
    coeffs_num_mn = [m, n]

    if simple_form == 'cancel':
        fraction_ret1 = sp.cancel(variable1 / Poly(coeffs_num_mn, variable2))
    else:
        fraction_ret1 = sp.factor(variable1 / Poly(coeffs_num_mn, variable2))

    if simple_form == 'cancel':
        fraction_ret2 = sp.cancel(Poly(coeffs_num_kl, variable2) / Poly(coeffs_num_mn, variable2))
    else:
        fraction_ret2 = sp.factor(Poly(coeffs_num_kl, variable2) / Poly(coeffs_num_mn, variable2))

    if variable2 and variable1 in fraction_ret1.free_symbols:
        return fraction_ret1, fraction_ret2
    else:
        return generate_random_args(variable1, variable2, simple_form=simple_form)    
    
#利用action打乱arg1,arg2roj
def act_arg (arg1, arg2, action_name, simple_form='cancel'):
    #with given action
    if action_name=='S':
        if simple_form=='cancel':
            return sp.cancel(arg1/arg2),sp.cancel(-1/arg2)
        else:
            return sp.factor(arg1/arg2),sp.factor(-1/arg2)
        
    elif action_name=='T':
        if simple_form=='cancel':
            return sp.cancel(arg1),sp.cancel(arg2+1)
        else:
            return sp.factor(arg1),sp.factor(arg2+1)
        
    elif action_name=='no':
        if simple_form=='cancel':
            return sp.cancel(arg1),sp.cancel(arg2)
        else:
            return sp.factor(arg1),sp.factor(arg2)        

    else:
        raise NameError('Acting on argument with {} is not implemented'.format(action_name))
    
def generate_action_list(action_num):
    action_list=[]
    for i in range(action_num):
        action_list.append(random.choice(['S', 'T', 'no']))

    return action_list 

def act_modular_random(arg1, arg2 , action_list):
    """act on arg1 and arg2 and return q_theta function after action in list"""
    new_arg1, new_arg2=arg1, arg2
    for action in action_list:
        #print('action {}' .format(action))
        old_arg1, old_arg2=new_arg1, new_arg2
        #print('Before doing action :Old arg1 ={}, old arg2 ={}'.format(old_arg1, old_arg2))
        new_arg1, new_arg2=act_arg (old_arg1, old_arg2, action_name=action, simple_form='cancel')
        #print('After doing action {}:new arg1={}, new arg2={}'.format(action,new_arg1,new_arg2))
        
    return q_theta(new_arg1, new_arg2)



'''(arg1, arg2)=generate_random_args(variable1=z,variable2=t)
print(q_theta(arg1, arg2))
action_num=5
action_list=generate_action_list(action_num)
print(action_list)
print(act_modular_random(arg1, arg2, action_list, action_num))'''


def scr_null_state(num_times_one, variable1, variable2):

    simple_form = 1
    expr_generated = 1

    #k can determine number of q_theta functions
    k=random.randint(1,3)
    for _ in range(k):
        (arg1, arg2)=generate_random_args(variable1, variable2)
        simple_term=q_theta(arg1, arg2)

        #训练的时候是3-5
        # action_num=random.randint(3, 5)
        
        #验证的时候是6-10
        action_num=random.randint(6, 10)

        action_list0=generate_action_list(action_num=action_num)

        complex_term=act_modular_random(arg1, arg2, action_list=action_list0)

        # n is judge number 
        n=random.randint(0,1)
        if n==0:
            simple_form*=simple_term
            expr_generated*=complex_term
        else:
            simple_form*=1/simple_term
            expr_generated*=1/complex_term

    for _ in range(num_times_one):
        #We will times one as term1/term2 =1 ,where both terms get scrambled
        
        (arg1, arg2)=generate_random_args(variable1,variable2)
        #训练的时候是3-5
        # action_num1, action_num2=random.randint(3, 5), random.randint(3, 5)

        #验证的时候是6-10
        action_num1, action_num2=random.randint(6, 10), random.randint(6, 10)

        action_list1=generate_action_list(action_num=action_num1)
        action_list2=generate_action_list(action_num=action_num2)
        term1=act_modular_random(arg1, arg2, action_list=action_list1)
        term2=act_modular_random(arg1, arg2, action_list=action_list2)
        expr_generated*=term1/term2

    if expr_generated==1:
        return scr_null_state(num_times_one, variable1, variable2)
    else:
        return simple_form, expr_generated


def generate_one_sample(seed):
    random.seed(seed)
    num_times_one = random.randint(1, 3)
    simple_form, expr_generated = scr_null_state(num_times_one, variable1=z, variable2=t)
    return (str(expr_generated), str(simple_form))


if __name__ == "__main__":

    n = 10000
    num_workers = 8  
    batch_size = 1000  
    
    print(f"使用 {num_workers} 个进程生成 {n} 条数据...")
    
    seeds = [random.randint(0, 2**31) for _ in range(n)]
    
    with open(f"./data/test/random_ns_nt/infix_origin.txt", 'w') as file_origin, \
         open("./data/test/random_ns_nt/infix_simple.txt", 'w') as file_simple:
        
        buffer_origin = []
        buffer_simple = []
        
        with Pool(num_workers) as pool:

            for i, (expr_generated, simple_form) in enumerate(
                tqdm.tqdm(pool.imap(generate_one_sample, seeds, chunksize=100), 
                         total=n, desc="Generating data")
            ):
                buffer_origin.append(f'{expr_generated}\n')
                buffer_simple.append(f'{simple_form}\n')
                
                if (i + 1) % batch_size == 0:
                    file_origin.writelines(buffer_origin)
                    file_simple.writelines(buffer_simple)
                    buffer_origin.clear()
                    buffer_simple.clear()
        
        if buffer_origin:
            file_origin.writelines(buffer_origin)
        if buffer_simple:
            file_simple.writelines(buffer_simple)
    
    print("数据生成完成！")


