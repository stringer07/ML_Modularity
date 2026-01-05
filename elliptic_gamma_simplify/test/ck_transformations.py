import mpmath as mp
import random

# ==========================================
# 1. 数值计算引擎 (支持全复平面参数)
# ==========================================
mp.dps = 30  # 设置高精度

def theta0_safe(z, tau):
    """计算 theta0(z; tau) = (x; q)_inf * (q/x; q)_inf"""
    x = mp.exp(2 * mp.pi * 1j * z)
    q = mp.exp(2 * mp.pi * 1j * tau)
    return mp.qp(x, q) * mp.qp(q/x, q)

def elliptic_gamma_function(z, tau, sigma, max_terms=2500, tol=1e-20):
    z = mp.mpc(z)
    tau = mp.mpc(tau)
    sigma = mp.mpc(sigma)
    
    # --- 关键：利用公式扩展定义域到下半平面 ---
    # 如果 Im(tau) < 0，利用 G(z; -t, s) = 1 / G(z-t; -t, s) 的逆形式
    # 即 G(z; t, s) = 1 / G(z - t_real; -t_real, s) 其中 t_real = -t
    if mp.im(tau) < -1e-15:
        return 1 / elliptic_gamma_function(z - tau, -tau, sigma, max_terms, tol)
    if mp.im(sigma) < -1e-15:
        return 1 / elliptic_gamma_function(z - sigma, tau, -sigma, max_terms, tol)

    # 边界检查
    if abs(mp.im(tau)) < 1e-15 or abs(mp.im(sigma)) < 1e-15:
        return mp.mpc('nan')

    # 级数计算核心逻辑
    width_limit = abs(mp.im(tau)) + abs(mp.im(sigma))
    current_dist = mp.im(2*z - tau - sigma)
    
    # 在收敛域内直接计算
    if abs(current_dist) < width_limit * 0.99:
        sum_result = mp.mpc(0)
        A = mp.pi * (2*z - tau - sigma)
        B = mp.pi * tau
        C = mp.pi * sigma
        
        for j in range(1, max_terms + 1):
            j_mp = mp.mpf(j)
            sin_B = mp.sin(j_mp * B)
            sin_C = mp.sin(j_mp * C)
            if abs(sin_B) < 1e-25 or abs(sin_C) < 1e-25: continue
                
            term = mp.sin(j_mp * A) / (j_mp * sin_B * sin_C)
            sum_result += term
            if abs(term) < tol: break
        return mp.exp(-0.5j * sum_result)

    # 递归移位以进入收敛域
    if current_dist > 0:
        return theta0_safe(z - sigma, tau) * elliptic_gamma_function(z - sigma, tau, sigma, max_terms, tol)
    else:
        return elliptic_gamma_function(z + sigma, tau, sigma, max_terms, tol) / theta0_safe(z, tau)

# 简写调用
def egamma(z, t, s):
    return elliptic_gamma_function(z, t, s)

# ==========================================
# 2. 变换逻辑验证器 (严格复刻您的 act_func)
# ==========================================
def verify_user_transforms():
    print(f"{'Transform Name':<20} | {'Status':<15} | {'Error (Diff)':<15} | {'Comment'}")
    print("-" * 80)

    # 生成随机测试参数 (保证初始参数在上半平面)
    t = mp.mpc(complex(random.uniform(-0.5, 0.5), random.uniform(0.3, 0.8)))
    s = mp.mpc(complex(random.uniform(-0.5, 0.5), random.uniform(0.3, 0.8)))
    z = mp.mpc(complex(random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)))
    
    # 计算基准值 LHS
    lhs = egamma(z, t, s)

    # 定义所有变换及其对应的 RHS 计算逻辑
    transforms = {
        'none': lambda: egamma(z, t, s),
        
        'symmetry': lambda: egamma(z, s, t),
        
        'periodicity_z': lambda: egamma(z + 1, t, s),
        'periodicity_t': lambda: egamma(z, t + 1, s),
        'periodicity_s': lambda: egamma(z, t, s + 1),
        
        'inversion': lambda: 1 / egamma(t + s - z, t, s),
        
        'shift1': lambda: 1 / egamma(z - t, -t, s),
        'shift2': lambda: egamma(s - z, -t, s),
        
        'mod1': lambda: egamma(z, t - s, s) * egamma(z, s - t, t),
        
        'mod2': lambda: egamma(z/s, t/s, -1/s) * egamma(z/t, s/t, -1/t),
        
        'dup': lambda: (
            egamma(z/2, t, s) *
            egamma(z/2 + t/2, t, s) *
            egamma(z/2 + s/2, t, s) *
            egamma(z/2 + t/2 + s/2, t, s) *
            egamma(z/2 + 0.5, t, s) *
            egamma(z/2 + t/2 + 0.5, t, s) *
            egamma(z/2 + s/2 + 0.5, t, s) *
            egamma(z/2 + t/2 + s/2 + 0.5, t, s)
        )
    }

    # 循环验证
    for name, rhs_func in transforms.items():
        try:
            rhs = rhs_func()
            
            # 1. 计算复数差异
            diff = abs(lhs - rhs)
            
            # 2. 计算模长差异 (专门针对 mod2)
            mod_diff = abs(abs(lhs) - abs(rhs))
            
            if diff < 1e-10:
                status = "✅ PASS"
                comment = "Exact Match"
            elif mod_diff < 1e-10:
                status = "⚠️ PARTIAL"
                comment = "Modulus Match (Phase differs)"
            else:
                status = "❌ FAIL"
                comment = "Mismatch"
                
            print(
                f"{name:<20} | {status:<15} | {float(diff):.2e}".ljust(53)
                + f" | {comment}"
            )

            
        except Exception as e:
            print(f"{name:<20} | ❌ ERROR       | N/A             | {str(e)}")

# ==========================================
# 3. 执行
# ==========================================
if __name__ == "__main__":
    print("Running verification on `act_func` logic...")
    # 运行多次以防随机巧合
    for i in range(1, 3):
        print(f"\n--- Test Batch {i} ---")
        verify_user_transforms()