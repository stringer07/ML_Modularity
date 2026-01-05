import time
import sys
import multiprocessing
from multiprocessing import Pool, Queue, Process
from queue import Empty
from pathlib import Path
import random
import numpy as np
import sympy as sp
import mpmath as mp
import torch
import functools
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration

from src.utils import AttrDict
from src.envs import build_env

class GlobalConfig:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 32            
    NUM_WORKERS_PREPROCESS = 12    
    NUM_WORKERS_VERIFY = 20        
    QUEUE_TIMEOUT = 5              
    
    MODEL_PATH = "./ckpoints/model_for_inference"
    TOKENIZER_PATH = "./ckpoints/model_for_inference"
    
    DATA_DIR = None 
    INPUT_FILE = None
    TARGET_FILE = None
    
    LOG_FILE = "eval_result.txt"
    
    TEST_SAMPLE_LIMIT = 5000 
    
    MP_DPS = 30
    H_STEP = mp.mpf("0.07")
    INVARIANT_THRESHOLD = 1e-3
    MAX_TS_RETRIES = 5
    
    NUM_Z_POINTS = 6
    MAX_Z_TRIALS = 100  
    
    ENV_PARAMS = AttrDict({
        'env_name': 'char_sp',
        'int_base': 10,
        'balanced': False,
        'positive': True,
        'precision': 10,
        'n_variables': 1,
        'n_coefficients': 0,
        'leaf_probs': '0.75,0,0.25,0',
        'max_len': 1000,
        'max_int': 5,
        'max_ops': 15,
        'max_ops_G': 15,
        'clean_prefix_expr': True,
        'rewrite_functions': '',
        'tasks': 'prim_fwd',
        'operators': 'add:10,sub:3,mul:10,div:5,sqrt:4,pow2:4,pow3:2,pow4:1,pow5:1,ln:4,exp:4,sin:4,cos:4,tan:4,asin:1,acos:1,atan:1,sinh:1,cosh:1,tanh:1,asinh:1,acosh:1,atanh:1',
    })


class Logger:
    def __init__(self, filename, init_msg=True):
        self.filename = filename
        if init_msg:
            with open(self.filename, 'a', encoding='utf-8') as f:
                f.write(f"\n\n{'#'*60}\n")
                f.write(f"=== Batch Evaluation Session Started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                f.write(f"{'#'*60}\n\n")

    def log(self, msg, to_console=True):
        if to_console:
            print(msg)
        with open(self.filename, 'a', encoding='utf-8') as f:
            f.write(msg + "\n")

logger = Logger(GlobalConfig.LOG_FILE)


@functools.lru_cache(maxsize=128)
def theta0_safe(z, tau):
    x = mp.exp(2 * mp.pi * 1j * z)
    q = mp.exp(2 * mp.pi * 1j * tau)
    return mp.qp(x, q) * mp.qp(q/x, q)

@functools.lru_cache(maxsize=256)
def elliptic_gamma_function(z, tau, sigma, max_terms=3000, depth=0):
    if depth > 50:
        return mp.mpc('nan')

    dynamic_tol = mp.eps * 100
    
    im_tau = mp.im(tau)
    if im_tau < -dynamic_tol:
        return 1 / elliptic_gamma_function(z - tau, -tau, sigma, max_terms, depth + 1)
    
    im_sigma = mp.im(sigma)
    if im_sigma < -dynamic_tol:
        return 1 / elliptic_gamma_function(z - sigma, tau, -sigma, max_terms, depth + 1)

    if abs(im_tau) < dynamic_tol or abs(im_sigma) < dynamic_tol:
        return mp.mpc('nan')

    width_limit = abs(im_tau) + abs(im_sigma)
    current_dist = mp.im(2*z) - im_tau - im_sigma
    
    if abs(current_dist) < width_limit * 0.99:
        sum_result = mp.mpc(0)
        pi_val = mp.pi
        A = pi_val * (2*z - tau - sigma)
        B = pi_val * tau
        C = pi_val * sigma
        
        for j in range(1, max_terms + 1):
            sin_B = mp.sin(j * B)
            sin_C = mp.sin(j * C)
            if abs(sin_B) < mp.eps or abs(sin_C) < mp.eps: continue
            
            term = mp.sin(j * A) / (j * sin_B * sin_C)
            sum_result += term
            if abs(term) < dynamic_tol: break
            
        return mp.exp(-0.5j * sum_result)

    if current_dist > 0:
        shifted_val = elliptic_gamma_function(z - sigma, tau, sigma, max_terms, depth + 1)
        th0_val = theta0_safe(z - sigma, tau)
        return th0_val * shifted_val
    else:
        shifted_val = elliptic_gamma_function(z + sigma, tau, sigma, max_terms, depth + 1)
        th0_val = theta0_safe(z, tau)
        return shifted_val / th0_val

def egamma(z, t, s):
    return elliptic_gamma_function(z, t, s)

def eval_expr_val(expr_code, z_val, t_val, s_val):
    """计算表达式值，输入为编译后的 Code Object"""
    context = {
        "egamma": egamma,
        "z": z_val,
        "t": t_val,
        "s": s_val,
        "mp": mp,
    }
    try:
        val = eval(expr_code, {"__builtins__": None}, context)
        if not mp.isfinite(val) or val == 0:
            return None
        return val
    except Exception:
        return None

def fourth_diff_invariant(R_vals):
    try:
        return (
            R_vals[0] * R_vals[4]
            / (R_vals[1]**4 * R_vals[3]**4)
            * (R_vals[2]**6)
        )
    except Exception:
        return None

def find_simple_safe_points(origin_code, simple_code, num_z=None, max_trials=None, h=None):
    if num_z is None: num_z = GlobalConfig.NUM_Z_POINTS
    if max_trials is None: max_trials = GlobalConfig.MAX_Z_TRIALS
    if h is None: h = GlobalConfig.H_STEP

    for _ in range(50):  
        t_val = complex(random.uniform(-0.8, 0.8), random.uniform(0.2, 0.9))
        s_val = complex(random.uniform(-0.8, 0.8), random.uniform(0.2, 0.9))
        t_mp = mp.mpc(t_val)
        s_mp = mp.mpc(s_val)

        valid_z = []
        for _ in range(max_trials):
            z0 = complex(random.uniform(-0.6, 0.6), random.uniform(-0.6, 0.6))
            z0_mp = mp.mpc(z0)

            ok = True
            for k in range(5):
                z_k = z0_mp + k * h
                v1 = eval_expr_val(origin_code, z_k, t_mp, s_mp)
                v2 = eval_expr_val(simple_code, z_k, t_mp, s_mp)
                if v1 is None or v2 is None:
                    ok = False
                    break
            if ok:
                valid_z.append(z0)
            if len(valid_z) >= num_z:
                return t_val, s_val, valid_z
    return None, None, None

def check_at_z(origin_code, simple_code, z0, t_val, s_val, h):
    R_vals = []
    for k in range(5):
        z_k = z0 + k * h
        v_org = eval_expr_val(origin_code, z_k, t_val, s_val)
        v_sim = eval_expr_val(simple_code, z_k, t_val, s_val)
        if v_org is None or v_sim is None: return None

        r = v_org / v_sim
        if not mp.isfinite(r) or r == 0: return None
        R_vals.append(r)
    return fourth_diff_invariant(R_vals)



def preprocess_worker(worker_id, input_queue, output_queue, test_inputs, test_simples, env_params_dict):
    local_params = AttrDict(env_params_dict)
    local_env = build_env(local_params)
    if 'egamma' not in local_env.local_dict:
        local_env.local_dict['egamma'] = sp.Function('egamma')
    
    while True:
        try:
            task = input_queue.get(timeout=1)
            if task is None: break
            indices = task
            batch_data = []
            for n in indices:
                raw_input = test_inputs[n].strip()
                simple_target = test_simples[n].strip()
                try:
                    expr_sp = sp.S(raw_input, locals=local_env.local_dict)
                    expr_prefix = local_env.sympy_to_prefix(expr_sp)
                    if len(expr_prefix) <= GlobalConfig.ENV_PARAMS.max_len:
                        batch_data.append({'index': n, 'prefix': expr_prefix, 'simple': simple_target, 'status': 'success'})
                    else:
                        batch_data.append({'index': n, 'simple': simple_target, 'status': 'too_long'})
                except Exception as e:
                    batch_data.append({'index': n, 'simple': simple_target, 'status': 'error', 'error': str(e)})
            output_queue.put(batch_data)
        except Empty: continue
        except Exception: break

def verify_sample_worker(args):
    idx, simple_truth, prediction, env_params_dict = args
    mp.dps = GlobalConfig.MP_DPS

    TIMEOUT_LIMIT = 180
    start_time_worker = time.time()
    
    local_params = AttrDict(env_params_dict)
    local_env = build_env(local_params)
    if 'egamma' not in local_env.local_dict:
        local_env.local_dict['egamma'] = sp.Function('egamma')
        
    fourth_diff_pass = False
    sym_pass = False
    error_msg = None
    
    try:
        simple_truth_code = compile(simple_truth, '<string>', 'eval')
        prediction_code = compile(prediction, '<string>', 'eval')
    except Exception as e:
        return (idx, False, False, f"Compilation Error: {str(e)}")

    for attempt in range(GlobalConfig.MAX_TS_RETRIES):
        if time.time() - start_time_worker > TIMEOUT_LIMIT:
            error_msg = "Verification Timeout (Hard Limit)"
            break

        try:
            t_val, s_val, z_list = find_simple_safe_points(simple_truth_code, prediction_code)
        except Exception as e:
            if error_msg is None: error_msg = f"Find points error: {str(e)}"
            continue

        if t_val is None or not z_list:
            continue

        t_mp = mp.mpc(t_val)
        s_mp = mp.mpc(s_val)
        passed = True

        for z0 in z_list:

            if time.time() - start_time_worker > TIMEOUT_LIMIT:
                passed = False
                error_msg = "Verification Timeout (Inside Loop)"
                break
                
            z0_mp = mp.mpc(z0)
            I = check_at_z(simple_truth_code, prediction_code, z0_mp, t_mp, s_mp, GlobalConfig.H_STEP)

            if I is None or abs(I - 1) > GlobalConfig.INVARIANT_THRESHOLD:
                passed = False
                break

        if error_msg and "Timeout" in error_msg:
            break

        if passed:
            fourth_diff_pass = True
            break

    if not (error_msg and "Timeout" in error_msg):
        try:
            sp_pred = sp.S(prediction, locals=local_env.local_dict)
            sp_truth = sp.S(simple_truth, locals=local_env.local_dict)
            if sp.simplify(sp_pred - sp_truth) == 0:
                sym_pass = True
        except Exception:
            pass

    return (idx, fourth_diff_pass, sym_pass, error_msg)

def run_evaluation(dataset_dir):
    dataset_name = dataset_dir.name
    
    GlobalConfig.DATA_DIR = dataset_dir
    GlobalConfig.INPUT_FILE = dataset_dir / "infix_origin.txt"
    GlobalConfig.TARGET_FILE = dataset_dir / "infix_simple.txt"

    logger.log("\n" + "="*80)
    logger.log(f"PROCESSING DATASET: {dataset_name}")
    logger.log(f"PATH: {dataset_dir}")
    logger.log("="*80)

    if not GlobalConfig.INPUT_FILE.exists() or not GlobalConfig.TARGET_FILE.exists():
        logger.log(f"❌ Error: Files not found in {dataset_dir}. Skipping...")
        return

    logger.log(f"Loading data from {GlobalConfig.DATA_DIR}...")
    with open(GlobalConfig.INPUT_FILE, 'r') as f:
        test_inputs = f.readlines()
    with open(GlobalConfig.TARGET_FILE, 'r') as f:
        test_simples = f.readlines()
        
    if GlobalConfig.TEST_SAMPLE_LIMIT:
        test_inputs = test_inputs[:GlobalConfig.TEST_SAMPLE_LIMIT]
        test_simples = test_simples[:GlobalConfig.TEST_SAMPLE_LIMIT]
        logger.log(f"⚠️ Limit applied: {GlobalConfig.TEST_SAMPLE_LIMIT} samples")
        
    total_samples = len(test_inputs)
    logger.log(f"Total Samples: {total_samples}")

    logger.log(f"Loading Model on {GlobalConfig.DEVICE}...")
    tokenizer = T5Tokenizer.from_pretrained(GlobalConfig.TOKENIZER_PATH)
    model = T5ForConditionalGeneration.from_pretrained(GlobalConfig.MODEL_PATH)
    model = model.to(GlobalConfig.DEVICE).eval()
    
    env = build_env(AttrDict(dict(GlobalConfig.ENV_PARAMS)))

    logger.log("\n>>> Step 1: Preprocessing & Inference (Pipeline)...")
    
    input_queue = Queue(maxsize=GlobalConfig.NUM_WORKERS_PREPROCESS * 4)
    output_queue = Queue()
    
    workers = []
    safe_params_dict = dict(GlobalConfig.ENV_PARAMS)
    
    for i in range(GlobalConfig.NUM_WORKERS_PREPROCESS):
        p = Process(target=preprocess_worker, 
                    args=(i, input_queue, output_queue, test_inputs, test_simples, safe_params_dict))
        p.start()
        workers.append(p)

    batch_indices = []
    for n in range(total_samples):
        batch_indices.append(n)
        if len(batch_indices) == GlobalConfig.BATCH_SIZE:
            input_queue.put(batch_indices)
            batch_indices = []
    if batch_indices:
        input_queue.put(batch_indices)

    for _ in range(GlobalConfig.NUM_WORKERS_PREPROCESS):
        input_queue.put(None)
        
    predictions_to_verify = [] 
    
    total_batches = (total_samples + GlobalConfig.BATCH_SIZE - 1) // GlobalConfig.BATCH_SIZE
    processed_batches = 0
    empty_wait_count = 0
    
    pbar_inf = tqdm(total=total_batches, desc=f"Inferencing [{dataset_name}]", unit="batch")
    
    start_time_inf = time.time()
    
    while processed_batches < total_batches:
        try:
            batch_data = output_queue.get(timeout=GlobalConfig.QUEUE_TIMEOUT)
            empty_wait_count = 0 
            processed_batches += 1
            
            valid_prefixes = []
            valid_meta = []
            
            for item in batch_data:
                if item['status'] == 'success':
                    valid_prefixes.append(item['prefix'])
                    valid_meta.append(item)
            
            if valid_prefixes:
                with torch.no_grad():
                    inputs = tokenizer(valid_prefixes, return_tensors="pt", is_split_into_words=True, 
                                       padding=True, truncation=True)
                    inputs = {k: v.to(GlobalConfig.DEVICE) for k, v in inputs.items()}
                    outputs = model.generate(inputs['input_ids'], max_length=1024, 
                                             num_beams=2, early_stopping=True)
                    preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                
                for meta, pred_str in zip(valid_meta, preds):
                    try:
                        prefix_list = pred_str.split(" ")
                        infix = env.prefix_to_infix(prefix_list)
                        sp_obj = env.infix_to_sympy(infix)
                        final_pred_str = str(sp_obj)
                        predictions_to_verify.append((meta['index'], meta['simple'], final_pred_str))
                    except:
                        pass 
            pbar_inf.update(1)
        except Empty:
            empty_wait_count += 1
            if empty_wait_count > 20: 
                logger.log("⚠️ Warning: Queue timeout. Workers might be stuck.")
                break
            continue

    pbar_inf.close()
    for p in workers:
        p.join()
        
    time_inf = time.time() - start_time_inf
    logger.log(f"Inference Done. Time: {time_inf:.2f}s. Valid Outputs: {len(predictions_to_verify)}")
    
    del model, tokenizer
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    logger.log(f"\n>>> Step 2: Verification with Fourth-Diff Invariant (Parallel CPU: {GlobalConfig.NUM_WORKERS_VERIFY})...")

    verify_tasks = [
        (idx, simple_truth, pred, safe_params_dict) 
        for (idx, simple_truth, pred) in predictions_to_verify
    ]
    
    stats = {'fourth_diff': 0, 'sym': 0, 'both': 0, 'total': len(verify_tasks)}
    start_time_verify = time.time()
    
    if verify_tasks:
        with Pool(processes=GlobalConfig.NUM_WORKERS_VERIFY) as pool:
            # chunksize=1 配合 imap_unordered 有助于平滑进度条
            iterator = pool.imap_unordered(verify_sample_worker, verify_tasks, chunksize=1)
            pbar_ver = tqdm(total=len(verify_tasks), desc=f"Verifying [{dataset_name}]", unit="sample")
            
            for idx, fourth_pass, s_pass, err in iterator:
                if fourth_pass: stats['fourth_diff'] += 1
                if s_pass: stats['sym'] += 1
                if fourth_pass and s_pass: stats['both'] += 1
                
                current_total = pbar_ver.n + 1
                pbar_ver.set_postfix({
                    '4th-Diff': f"{stats['fourth_diff']/current_total:.2%}",
                    'Sym': f"{stats['sym']/current_total:.2%}"
                })
                pbar_ver.update(1)
            
            pbar_ver.close()
    else:
        logger.log("No valid predictions to verify.")

    time_verify = time.time() - start_time_verify
    valid_count = stats['total']
    
    logger.log("\n" + "-"*60)
    logger.log(f"REPORT FOR: {dataset_name}")
    logger.log("-"*60)
    logger.log(f"Total Input Samples:     {total_samples}")
    logger.log(f"Valid Predictions:       {valid_count}")
    
    if valid_count > 0:
        logger.log(f"Fourth-Diff Invariant:   {stats['fourth_diff']}/{valid_count} ({stats['fourth_diff']/valid_count:.2%})")
        logger.log(f"Symbolic Strict Acc:     {stats['sym']}/{valid_count} ({stats['sym']/valid_count:.2%})")
        logger.log(f"Both Agree Acc:          {stats['both']}/{valid_count} ({stats['both']/valid_count:.2%})")
    else:
        logger.log("No valid predictions generated.")
        
    logger.log("-" * 60)
    logger.log(f"Time Taken: Inference={time_inf:.2f}s, Verify={time_verify:.2f}s")
    logger.log("="*60 + "\n")
    
    import gc
    gc.collect()

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('fork', force=True)
    except RuntimeError:
        pass
    
    test_base_dir = Path("./data/test/ns7nt2")
    
    if not test_base_dir.exists():
        print(f"❌ Error: Base directory {test_base_dir} does not exist.")
        sys.exit(1)

    all_dirs = [d for d in test_base_dir.iterdir() if d.is_dir()]
    target_dirs = [d for d in all_dirs if not d.name.endswith("nodup")]
    target_dirs.sort(key=lambda x: x.name)
    
    print(f"\n🔍 Found {len(target_dirs)} datasets to process (excluding '*nodup').")
    print(f"📝 Logging results to: {GlobalConfig.LOG_FILE}")
    
    try:
        for idx, d in enumerate(target_dirs):
            print(f"\n🚀 [{idx+1}/{len(target_dirs)}] Starting evaluation for: {d.name}")
            run_evaluation(d)
            
        print("\n✅ All datasets processed successfully.")
        os.system("/usr/bin/shutdown")
            
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.log(f"\n❌ Critical Error in Main Process: {e}")
        import traceback
        traceback.print_exc()
        os.system("/usr/bin/shutdown")