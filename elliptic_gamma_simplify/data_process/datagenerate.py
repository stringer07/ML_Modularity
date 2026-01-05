import sympy as sp
from sympy import Poly
import random
from sympy.core.function import Function
import os
from tqdm import tqdm
import gzip
from multiprocessing import Pool, cpu_count
from functools import partial
import numpy as np


egamma=Function('egamma')
z=sp.Symbol('z')
t=sp.Symbol('t')
s=sp.Symbol('s')
  
#generate random elliptic egamma function pairs
def generate_random_egamma_v1(var1, var2, var3):
    """return elliptic gamma function pairs with random argument"""
    #generate k,l,m,n,k1,l1,n1
    while True:
        [k, l, n, k1, l1, n1] = [random.randint(-9, 9) for _ in range(6)]
        if l != 0 and l1 != 0:
            m = (k * n - 1) / l
            if m.is_integer() and -9 <= m <= 9 and m == (k1 * n1 - 1) / l1:
                m = int(m)
                break
        elif l == 0 and k * n == 1 and l1 != 0:
            m = (k1 * n1 - 1) / l1
            if m.is_integer() and -9 <= m <= 9:
                m = int(m)
                break
        elif l1 == 0 and k1 * n1 == 1 and l != 0:
            m = (k * n - 1) / l
            if m.is_integer() and -9 <= m <= 9:
                m = int(m)
                break
        elif l == 0 and l1 == 0 and k * n == 1 and k1 * n1 == 1:
            m = random.randint(-9, 9)
            break
    if (k*n-m*l) != 1 or (k1*n1-m*l1) != 1:
        raise ValueError("dosen't satisfy SL(3,Z), [k,l,m,n,k1,l1,m,n1]={}".format([k,l,m,n,k1,l1,m,n1]))


    denominator1=Poly([m,n],var3)
    denominator2=Poly([m,n1],var2)
    arg11=sp.cancel(Poly(var1)/denominator1)
    arg12=sp.cancel(Poly(var2-n1*(k*var3+l))/denominator1)
    arg13=sp.cancel(Poly([k, l],var3)/denominator1)
    arg21=sp.cancel(Poly(var1)/denominator2)
    arg22=sp.cancel(Poly(var3-n*(k1*var2+l1))/denominator2)
    arg23=sp.cancel(Poly([k1, l1],var2)/denominator2)
    
    exprs=[arg11,arg12,arg13,arg21,arg22,arg23]
    result = [expr.free_symbols.issubset({var1,var2,var3}) for expr in exprs]
    
    if all(result):
        return egamma(arg11,arg12,arg13)*egamma(arg21,arg22,arg23)
    else:
        return generate_random_egamma_v1(var1=z, var2=t, var3=s)
    
def generate_random_egamma_v2(var1, var2, var3):
    """return elliptic gamma function with random argument"""
    #generate k,l,m,n,k1,l1,n1
    while True:
        [k, l, n, k1, l1, n1] = [random.randint(-9, 9) for _ in range(6)]
        if l != 0 and l1 != 0:
            m = (k * n - 1) / l
            if m.is_integer() and -9 <= m <= 9 and m == (k1 * n1 - 1) / l1:
                m = int(m)
                break
        elif l == 0 and k * n == 1 and l1 != 0:
            m = (k1 * n1 - 1) / l1
            if m.is_integer() and -9 <= m <= 9:
                m = int(m)
                break
        elif l1 == 0 and k1 * n1 == 1 and l != 0:
            m = (k * n - 1) / l
            if m.is_integer() and -9 <= m <= 9:
                m = int(m)
                break
        elif l == 0 and l1 == 0 and k * n == 1 and k1 * n1 == 1:
            m = random.randint(-9, 9)
            break
    if k*n-m*l != 1 or k1*n1-m*l1 != 1:
        raise ValueError("dosen't satisfy SL(3,Z), [k,l,m,n,k1,l1,m,n1]={}".format([k,l,m,n,k1,l1,m,n1]))


    denominator1=Poly([m,n],var3)
    denominator2=Poly([m,n1],var2)
    arg11=sp.cancel(Poly(var1)/denominator1)
    arg12=sp.cancel(Poly(var2-n1*(k*var3+l))/denominator1)
    arg13=sp.cancel(Poly([k, l],var3)/denominator1)

    arg21=sp.cancel(Poly(var1)/denominator2)
    arg22=sp.cancel(Poly(var3-n*(k1*var2+l1))/denominator2)
    arg23=sp.cancel(Poly([k1, l1],var2)/denominator2)
    
    exprs=[arg11,arg12,arg13,arg21,arg22,arg23]
    result = [expr.free_symbols.issubset({var1,var2,var3}) for expr in exprs]
    
    if all(result):
        n = random.randint(0, 1)
        n_dup = random.randint(0, 1)
        if n==0 and n_dup==0:
            return egamma(arg11,arg12,arg13)
        elif n==0 and n_dup==1:
            return egamma(2*arg11,arg12,arg13)
        elif n==1 and n_dup==0:
            return egamma(arg21,arg22,arg23)
        elif n==1 and n_dup==1:
            return egamma(2*arg21,arg22,arg23)
    else:
        return generate_random_egamma_v2(var1=z, var2=t, var3=s)



#actions of elliptic egamma
def act_func (egamma_func, action_name):
    """actions of elliptic egamma"""

    if len(egamma_func.args)!=3:
        print("Wrong egamma function arguments: {}".format(egamma_func))
        return egamma_func
    else:
        (arg1, arg2, arg3)=egamma_func.args

    if action_name=='symmetry':
        return egamma(sp.cancel(arg1),sp.cancel(arg3),sp.cancel(arg2))
    
    elif action_name=='periodicity_z':
        return egamma(sp.cancel(arg1+1),sp.cancel(arg2),sp.cancel(arg3))
    elif action_name=='periodicity_t':
        return egamma(sp.cancel(arg1),sp.cancel(arg2+1),sp.cancel(arg3))
    elif action_name=='periodicity_s':
        return egamma(sp.cancel(arg1),sp.cancel(arg2),sp.cancel(arg3+1))
    
    elif action_name=='inversion':
        return 1/egamma(sp.cancel(arg2+arg3-arg1),sp.cancel(arg2),sp.cancel(arg3))
    
    elif action_name=='shift1':
        return 1/egamma(sp.cancel(arg1-arg2),sp.cancel(-arg2),sp.cancel(arg3))
    elif action_name=='shift2':
        return egamma(sp.cancel(arg3-arg1),sp.cancel(-arg2),sp.cancel(arg3))
    
    elif action_name=='mod1':
        term1=egamma(sp.cancel(arg1),sp.cancel(arg2-arg3),sp.cancel(arg3))
        term2=egamma(sp.cancel(arg1),sp.cancel(arg3-arg2),sp.cancel(arg2))
        return term1*term2
    elif action_name=='mod2':
        term1=egamma(sp.cancel(arg1/arg3),sp.cancel(arg2/arg3),sp.cancel(-1/arg3))
        term2=egamma(sp.cancel(arg1/arg2),sp.cancel(arg3/arg2),sp.cancel(-1/arg2))
        return term1*term2
    
    elif action_name=='dup':
        term1=egamma(sp.cancel(arg1/2),sp.cancel(arg2),sp.cancel(arg3))
        term2=egamma(sp.cancel(arg1/2 + arg2/2),sp.cancel(arg2),sp.cancel(arg3))
        term3=egamma(sp.cancel(arg1/2 + arg3/2),sp.cancel(arg2),sp.cancel(arg3))
        term4=egamma(sp.cancel(arg1/2 + arg2/2 + arg3/2),sp.cancel(arg2),sp.cancel(arg3))
        term5=egamma(sp.cancel(arg1/2 + sp.Rational(1, 2)),sp.cancel(arg2),sp.cancel(arg3))
        term6=egamma(sp.cancel(arg1/2 + arg2/2 + sp.Rational(1, 2)),sp.cancel(arg2),sp.cancel(arg3))
        term7=egamma(sp.cancel(arg1/2 + arg3/2 + sp.Rational(1, 2)),sp.cancel(arg2),sp.cancel(arg3))
        term8=egamma(sp.cancel(arg1/2 + arg2/2 + arg3/2 + sp.Rational(1, 2)),sp.cancel(arg2),sp.cancel(arg3))
        return term1*term2*term3*term4*term5*term6*term7*term8

    elif action_name=='none':
        return egamma(sp.cancel(arg1),sp.cancel(arg2),sp.cancel(arg3))
    
    else:
        raise NameError('Acting on argument with {} is not implemented'.format(action_name))
    


def generate_action_list_v1(action_num):
    action_list=[]
    act_list_full=['none','symmetry','periodicity_z','periodicity_t','periodicity_s','inversion','shift1','shift2','mod1','mod2']
    for _ in range(action_num):
        action_list.append(random.choice(act_list_full))

    return action_list

def generate_action_list_v2(action_num):
    action_list = []
    has_dup = False
    act_list_full=['none', 'symmetry','periodicity_z','periodicity_t','periodicity_s','inversion','shift1','shift2','mod1','mod2','dup']
    for _ in range(action_num):
        if not has_dup:
            action = random.choice(act_list_full)
            if action == 'dup':
                has_dup = True
        else:
            action = random.choice(act_list_full[:-1])
        action_list.append(action)

    return action_list


def count_egamma(expr, target_func=egamma):
    count = 0
    if isinstance(expr, sp.Pow) and isinstance(expr.base, sp.Function) and expr.base.func == target_func:
        count += abs(expr.exp)
    elif isinstance(expr, sp.Function) and expr.func == target_func:
        count += 1
    elif isinstance(expr, sp.Mul): 
        for factor in expr.args:
            count += count_egamma(factor, target_func)
    else:
        for arg in expr.args:
            count += count_egamma(arg, target_func)
    return count


def act(egamma_expr, action_list):
    """act on expression and return expression after action"""
    new_func = egamma_expr

    for action in action_list:
        #print('action: {}'.format(action))
        old_func=new_func
        #print('before action, func: {}'.format(old_func))
        if count_egamma(old_func)==1:
            numerator, denominator = old_func.as_numer_denom()
            if numerator!=1:
                new_func=act_func(numerator,action_name=action)
            else:
                new_func=1/act_func(denominator,action_name=action)
        else:
            func_list=list(old_func.args)
            old_element=random.choice(func_list)
            numerator, denominator=old_element.as_numer_denom()
            if numerator!=1:
                new_func=old_func.subs(numerator, act_func(numerator,action_name=action))
            else:
                new_func=old_func.subs(denominator, act_func(denominator,action_name=action))
        #print('after action, func: {}'.format(new_func))
    
    return new_func

def construct_pairs_v4(num_scramb, num_times_one, var1, var2, var3):
    simple_form = 1
    numerator_factors = []
    denominator_factors = []
    
    k = random.randint(1, 3)
    
    if k > 0:
        for _ in range(k):
            simple_term = generate_random_egamma_v2(var1, var2, var3)
            n = random.randint(0, 1)
            if n == 0:
                simple_form *= simple_term
                numerator_factors.append(simple_term)
            else:
                simple_form /= simple_term
                denominator_factors.append(simple_term)
    
    if num_times_one >= 1:
        for _ in range(num_times_one):
            egamma_func = generate_random_egamma_v2(var1, var2, var3)
            numerator_factors.append(egamma_func)
            denominator_factors.append(egamma_func)
    
    if len(numerator_factors) == 0 and len(denominator_factors) == 0:
        return construct_pairs_v4(num_scramb, num_times_one, var1, var2, var3)
    
    all_factors = [(f, True) for f in numerator_factors] + [(f, False) for f in denominator_factors]

    action_list = generate_action_list_v2(action_num=num_scramb)
    # for another test, without duplication
    # action_list = generate_action_list_v1(action_num=num_scramb)
    for action in action_list:
        
        idx = random.randint(0, len(all_factors) - 1)
        old_factor, is_numerator = all_factors[idx]
        
        try:
            new_factor = act(old_factor, [action])
            all_factors[idx] = (new_factor, is_numerator)
        except Exception as e:
            pass
            # print("old_factor: ", old_factor)
            # print("action: ", action)
            # print("e: ", e)
            
    expr_generated = 1
    for factor, is_numerator in all_factors:
        if is_numerator:
            expr_generated *= factor
        else:
            expr_generated /= factor
    
    if expr_generated == 1:
        return construct_pairs_v4(num_scramb, num_times_one, var1, var2, var3)
    
    return simple_form, expr_generated


def generate_single_pair(args):
    idx, seed = args
    random.seed(seed)
    

    num_scramb = random.randint(2, 5)
    num_times_one = random.randint(0, 2)
    
    try:
        simple_form, expr_generated = construct_pairs_v4(num_scramb, num_times_one, z, t, s)
        return (idx, str(simple_form), str(expr_generated))
    except Exception as e:
        try:
            simple_form, expr_generated = construct_pairs_v4(num_scramb, num_times_one, z, t, s)
            return (idx, str(simple_form), str(expr_generated))
        except:
            return None


def generate_batch(batch_args):
    start_idx, batch_size, base_seed = batch_args
    results = []
    
    random.seed(base_seed + start_idx)
    
    for i in range(batch_size):
        # for train set ns \in [3, 6]
        # num_scramb = random.randint(3, 6)
        
        # for another test set ns \in [7,10]
        num_scramb = random.randint(7, 10)
        
        num_times_one = random.randint(0, 2)
        # num_times_one = 3
        
        try:
            simple_form, expr_generated = construct_pairs_v4(num_scramb, num_times_one, z, t, s)
            results.append((start_idx + i, str(simple_form), str(expr_generated)))
        except Exception as e:
            try:
                simple_form, expr_generated = construct_pairs_v4(num_scramb, num_times_one, z, t, s)
                results.append((start_idx + i, str(simple_form), str(expr_generated)))
            except:
                pass
    
    return results


def parallel_generate(n, num_workers=8, batch_size=1000):
    batches = []
    base_seed = random.randint(0, 1000000)
    
    for start_idx in range(0, n, batch_size):
        actual_batch_size = min(batch_size, n - start_idx)
        batches.append((start_idx, actual_batch_size, base_seed))
    
    all_results = []
    
    with Pool(processes=num_workers) as pool:
        for batch_results in tqdm(pool.imap(generate_batch, batches), 
                                   total=len(batches), 
                                   desc="Generating data (parallel)"):
            all_results.extend(batch_results)
    
    all_results.sort(key=lambda x: x[0])
    
    return all_results




if __name__ == '__main__':
    n = 1000000
    num_workers = 20  

    data_type='random_nsnt'

    folder_path = './data/train/{}'.format(data_type)
    os.makedirs(folder_path, exist_ok=True)

    print(f"使用 {num_workers} 个CPU核心并行生成 {n} 个数据对...")
    
    results = parallel_generate(n, num_workers=num_workers, batch_size=200)
    
    print(f"生成完成，共 {len(results)} 个数据对，正在写入文件...")
    
    file1_path = os.path.join(folder_path, "infix_simple.txt")
    file2_path = os.path.join(folder_path, "infix_origin.txt")

    with open(file1_path, 'w') as fl1,\
          open(file2_path, 'w') as fl2:
        
        buffer_size = 200
        simple_buffer = []
        origin_buffer = []
        
        for i, (idx, simple_str, origin_str) in enumerate(tqdm(results, desc="Writing to files")):
            simple_buffer.append(f'{simple_str}\n')
            origin_buffer.append(f'{origin_str}\n')
            
            if (i + 1) % buffer_size == 0:
                fl1.writelines(simple_buffer)
                fl2.writelines(origin_buffer)
                simple_buffer.clear()
                origin_buffer.clear()
        
        if simple_buffer:
            fl1.writelines(simple_buffer)
            fl2.writelines(origin_buffer)
    
    print("完成！")



