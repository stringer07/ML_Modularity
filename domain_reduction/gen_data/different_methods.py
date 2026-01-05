import numpy as np
from fractions import Fraction
import torch
from decimal import Decimal, getcontext
from tqdm import tqdm

# 设置高精度计算
getcontext().prec = 50

# ==================== 采样方法 ====================

def pick_hyperbolic_coords(R, n_points):
    """
    在双曲极坐标下生成 n_points 个随机点。
    """
    theta = 2 * np.pi * np.random.rand(n_points)
    x = (np.cosh(R) - 1) * np.random.rand(n_points)
    r = np.arccosh(x + 1)
    return r, theta


def sample_disk_euclidean(n_points, radius=1.0):
    """
    在半径为 radius 的圆盘内，进行严格的【欧几里得】均匀采样。
    """
    theta = 2 * np.pi * np.random.rand(n_points)
    u_norm = np.random.rand(n_points)
    r = radius * np.sqrt(u_norm)
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x + 1j * y


def sample_disk_midchord(n_points, radius=1.0):
    """
    弦中点采样法：在圆周上随机取两点，返回弦的中点。
    """
    theta1 = 2 * np.pi * np.random.rand(n_points)
    theta2 = 2 * np.pi * np.random.rand(n_points)
    
    z1 = radius * np.exp(1j * theta1)
    z2 = radius * np.exp(1j * theta2)
    
    z_mid = (z1 + z2) / 2.0
    return z_mid


def sample_disk_uniform_radius(n_points, radius=1.0):
    """
    半径线性均匀采样：
    - theta ~ Uniform[0, 2π)
    - r ~ Uniform[0, radius)
    返回圆盘内点 z = r e^{i theta}
    """
    theta = 2 * np.pi * np.random.rand(n_points)
    r = radius * np.random.rand(n_points)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x + 1j * y


def map_disk_to_hp(z_disk):
    """
    将庞加莱圆盘模型的点映射到上半平面模型。
    """
    return 1j * (1 + z_disk) / (1 - z_disk)


def convert_to_half_plane_hyperbolic(r, theta):
    """
    将双曲极坐标点转换为上半平面模型中的复数坐标点。
    """
    z_disk = np.tanh(r / 2) * np.exp(1j * theta)
    z_hp = 1j * (1 + z_disk) / (1 - z_disk)
    return z_hp


# ==================== 基本域判断 ====================

def is_in_fundamental_domain(z):
    """
    检查一个或多个复数点是否位于标准基本域 F 内。
    """
    real_part = z.real
    modulus = np.abs(z)
    return (-0.5 <= real_part) & (real_part < 0.5) & (modulus >= 1)


# ==================== 数值转换 ====================

def decimal_to_fraction(decimal_value, precision=5):
    """
    将小数转换为最简分数形式。
    """
    rounded_value = round(decimal_value, precision)
    decimal_obj = Decimal(str(rounded_value))
    fraction = Fraction(decimal_obj)
    return fraction


def complex_to_fraction_tuple(complex_number, precision=5):
    """
    将复数转换为(实部分数, 虚部分数)的元组形式。
    """
    real_fraction = decimal_to_fraction(complex_number.real, precision)
    imag_fraction = decimal_to_fraction(complex_number.imag, precision)
    return (real_fraction, imag_fraction)


# ==================== 矩阵变换 ====================

def apply_matrix(a, b, c, d, z):
    """
    应用矩阵变换到复数 z = x + yi
    变换: (az + b) / (cz + d)
    """
    x, y = z
    num_re = a * x + b
    num_im = a * y
    den_re = c * x + d
    den_im = c * y
    denom = den_re * den_re + den_im * den_im
    z_re = (num_re * den_re + num_im * den_im) / denom
    z_im = (num_im * den_re - num_re * den_im) / denom
    return (z_re, z_im)


def reduce_to_fundamental_domain(z, max_steps=1000):
    """
    将复数点规约到基本域内。
    
    返回:
    (规约后的点, 变换矩阵)
    """
    a, b, c, d = 1, 0, 0, 1
    x, y = float(z[0]), float(z[1])
    steps = 0
    
    # 检查点是否有效（避免原点或虚部为0的点）
    if abs(y) < 1e-10 or (abs(x) < 1e-10 and abs(y) < 1e-10):
        return None, None
    
    # 平移操作：将实部移动到 [-0.5, 0.5) 区间
    while abs(x) >= 0.5:
        if steps > max_steps:
            return None, None
        n = int(round(x))
        b -= n * d
        a -= n * c
        x -= n
        steps += 1
    
    # 反演操作：如果模长小于1，进行反演
    while x*x + y*y < 1:
        if steps > max_steps:
            return None, None
        
        # 计算模长平方，避免除零
        mod_sq = x*x + y*y
        if mod_sq < 1e-10:  # 如果太接近原点，放弃这个点
            return None, None
        
        z = (-x/mod_sq, y/mod_sq)
        a, b, c, d = c, d, -a, -b
        x, y = float(z[0]), float(z[1])
        
        while abs(x) >= 0.5:
            if steps > max_steps:
                return None, None
            n = int(round(x))
            b -= n * d
            a -= n * c
            x -= n
            steps += 1
        steps += 1
    
    return (x, y), (a, b, c, d)


# ==================== 数据生成（统一接口） ====================

def generate_data_hyperbolic(num_samples, R_hyperbolic, precision=10):
    """
    使用双曲坐标生成训练数据。
    """
    print(f"\n{'='*60}")
    print(f"生成双曲测度数据 (R={R_hyperbolic})")
    print(f"{'='*60}")
    
    xs, ys, zs_prime = [], [], []
    count = 0
    batch_size = 10000
    
    pbar = tqdm(total=num_samples, desc=f"Hyperbolic R={R_hyperbolic}")
    
    while count < num_samples:
        # 生成一批双曲坐标点
        r_coords, theta_coords = pick_hyperbolic_coords(R_hyperbolic, batch_size)
        z_candidates = convert_to_half_plane_hyperbolic(r_coords, theta_coords)
        
        # 只保留基本域外的点
        mask_outside = ~is_in_fundamental_domain(z_candidates)
        outside_candidates = z_candidates[mask_outside]
        
        for z_complex in outside_candidates:
            if count >= num_samples:
                break
            
            if z_complex.imag <= 0:
                continue
            
            # 转换为分数形式
            z_frac = complex_to_fraction_tuple(z_complex, precision)
            
            # 规约到基本域
            z_prime, A = reduce_to_fundamental_domain(z_frac)
            
            if z_prime is None:
                continue
            
            a, b, c, d = A
            z_prime_exact = apply_matrix(a, b, c, d, (float(z_frac[0]), float(z_frac[1])))
            
            xs.append([z_frac[0].numerator, z_frac[0].denominator, 
                       z_frac[1].numerator, z_frac[1].denominator])
            ys.append(A)
            zs_prime.append([z_prime_exact[0], z_prime_exact[1]])
            
            count += 1
            pbar.update(1)
            
            if count >= num_samples:
                break
    
    pbar.close()
    
    X_tensor = torch.tensor(xs, dtype=torch.long)
    Y_tensor = torch.tensor(ys, dtype=torch.long)
    Zp_tensor = torch.tensor(zs_prime, dtype=torch.float)
    
    return X_tensor, Y_tensor, Zp_tensor


def generate_data_euclidean(num_samples, precision=10):
    """
    使用欧几里得均匀采样生成训练数据。
    """
    print(f"\n{'='*60}")
    print(f"生成欧几里得测度数据")
    print(f"{'='*60}")
    
    xs, ys, zs_prime = [], [], []
    count = 0
    batch_size = 10000
    
    pbar = tqdm(total=num_samples, desc="Euclidean")
    
    while count < num_samples:
        # 在圆盘中欧几里得均匀采样
        z_disk = sample_disk_euclidean(batch_size, radius=1.0)
        
        # 映射到上半平面
        z_hp = map_disk_to_hp(z_disk)
        
        # 只保留基本域外的点
        mask_outside = ~is_in_fundamental_domain(z_hp)
        outside_candidates = z_hp[mask_outside]
        
        for z_complex in outside_candidates:
            if count >= num_samples:
                break
            
            if z_complex.imag <= 0:
                continue
            
            z_frac = complex_to_fraction_tuple(z_complex, precision)
            z_prime, A = reduce_to_fundamental_domain(z_frac)
            
            if z_prime is None:
                continue
            
            a, b, c, d = A
            z_prime_exact = apply_matrix(a, b, c, d, (float(z_frac[0]), float(z_frac[1])))
            
            xs.append([z_frac[0].numerator, z_frac[0].denominator, 
                       z_frac[1].numerator, z_frac[1].denominator])
            ys.append(A)
            zs_prime.append([z_prime_exact[0], z_prime_exact[1]])
            
            count += 1
            pbar.update(1)
            
            if count >= num_samples:
                break
    
    pbar.close()
    
    X_tensor = torch.tensor(xs, dtype=torch.long)
    Y_tensor = torch.tensor(ys, dtype=torch.long)
    Zp_tensor = torch.tensor(zs_prime, dtype=torch.float)
    
    return X_tensor, Y_tensor, Zp_tensor


def generate_data_midchord(num_samples, precision=10):
    """
    使用弦中点采样生成训练数据。
    """
    print(f"\n{'='*60}")
    print(f"生成弦中点采样数据")
    print(f"{'='*60}")
    
    xs, ys, zs_prime = [], [], []
    count = 0
    batch_size = 10000
    
    pbar = tqdm(total=num_samples, desc="Midchord")
    
    while count < num_samples:
        # 弦中点采样
        z_disk = sample_disk_midchord(batch_size, radius=1.0)
        
        # 映射到上半平面
        z_hp = map_disk_to_hp(z_disk)
        
        # 只保留基本域外的点
        mask_outside = ~is_in_fundamental_domain(z_hp)
        outside_candidates = z_hp[mask_outside]
        
        for z_complex in outside_candidates:
            if count >= num_samples:
                break
            
            if z_complex.imag <= 0:
                continue
            
            z_frac = complex_to_fraction_tuple(z_complex, precision)
            z_prime, A = reduce_to_fundamental_domain(z_frac)
            
            if z_prime is None:
                continue
            
            a, b, c, d = A
            z_prime_exact = apply_matrix(a, b, c, d, (float(z_frac[0]), float(z_frac[1])))
            
            xs.append([z_frac[0].numerator, z_frac[0].denominator, 
                       z_frac[1].numerator, z_frac[1].denominator])
            ys.append(A)
            zs_prime.append([z_prime_exact[0], z_prime_exact[1]])
            
            count += 1
            pbar.update(1)
            
            if count >= num_samples:
                break
    
    pbar.close()
    
    X_tensor = torch.tensor(xs, dtype=torch.long)
    Y_tensor = torch.tensor(ys, dtype=torch.long)
    Zp_tensor = torch.tensor(zs_prime, dtype=torch.float)
    
    return X_tensor, Y_tensor, Zp_tensor


def generate_data_uniform_radius(num_samples, precision=10):
    """
    使用半径线性均匀采样生成训练数据。
    - 按 r~U[0,1), theta~U[0,2π) 采样
    """
    print(f"\n{'='*60}")
    print(f"生成半径均匀采样数据")
    print(f"{'='*60}")
    
    xs, ys, zs_prime = [], [], []
    count = 0
    batch_size = 10000
    
    pbar = tqdm(total=num_samples, desc="UniformRadius")
    
    while count < num_samples:
        # 半径线性均匀采样
        z_disk = sample_disk_uniform_radius(batch_size, radius=1.0)
        
        # 映射到上半平面
        z_hp = map_disk_to_hp(z_disk)
        
        # 只保留基本域外的点
        mask_outside = ~is_in_fundamental_domain(z_hp)
        outside_candidates = z_hp[mask_outside]
        
        for z_complex in outside_candidates:
            if count >= num_samples:
                break
            
            if z_complex.imag <= 0:
                continue
            
            z_frac = complex_to_fraction_tuple(z_complex, precision)
            z_prime, A = reduce_to_fundamental_domain(z_frac)
            
            if z_prime is None:
                continue
            
            a, b, c, d = A
            z_prime_exact = apply_matrix(a, b, c, d, (float(z_frac[0]), float(z_frac[1])))
            
            xs.append([z_frac[0].numerator, z_frac[0].denominator, 
                       z_frac[1].numerator, z_frac[1].denominator])
            ys.append(A)
            zs_prime.append([z_prime_exact[0], z_prime_exact[1]])
            
            count += 1
            pbar.update(1)
            
            if count >= num_samples:
                break
    
    pbar.close()
    
    X_tensor = torch.tensor(xs, dtype=torch.long)
    Y_tensor = torch.tensor(ys, dtype=torch.long)
    Zp_tensor = torch.tensor(zs_prime, dtype=torch.float)
    
    return X_tensor, Y_tensor, Zp_tensor


# ==================== 主程序 ====================

def verify_sample(X, Y, Zp, idx):
    """验证单个样本的正确性"""
    real_frac = Fraction(int(X[idx][0]), int(X[idx][1]))
    imag_frac = Fraction(int(X[idx][2]), int(X[idx][3]))
    z_original = (float(real_frac), float(imag_frac))
    
    a, b, c, d = [int(x) for x in Y[idx]]
    z_computed = apply_matrix(a, b, c, d, z_original)
    z_stored = Zp[idx].tolist()
    
    print(f"样本 {idx}:")
    print(f"  原始: {z_original[0]:.6f} + {z_original[1]:.6f}i")
    print(f"  矩阵: [[{a}, {b}], [{c}, {d}]]")
    print(f"  计算: {z_computed[0]:.6f} + {z_computed[1]:.6f}i")
    print(f"  存储: {z_stored[0]:.6f} + {z_stored[1]:.6f}i")
    print(f"  误差: {abs(z_computed[0] - z_stored[0]):.2e}, {abs(z_computed[1] - z_stored[1]):.2e}\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("统一数据集生成器")
    print("="*60)
    
    PRECISION = 5
    
    # 1. 生成双曲测度数据 R=10
    X_hyp10, Y_hyp10, Zp_hyp10 = generate_data_hyperbolic(
        num_samples=10000,
        R_hyperbolic=10,
        precision=PRECISION
    )
    
    # 2. 生成双曲测度数据 R=2
    X_hyp2, Y_hyp2, Zp_hyp2 = generate_data_hyperbolic(
        num_samples=10000,
        R_hyperbolic=2,
        precision=PRECISION
    )
    
    # 3. 生成欧几里得测度数据
    X_euc, Y_euc, Zp_euc = generate_data_euclidean(
        num_samples=10000,
        precision=PRECISION
    )
    
    # 4. 生成弦中点采样数据
    X_chord, Y_chord, Zp_chord = generate_data_midchord(
        num_samples=10000,
        precision=PRECISION
    )
    
    #5. 生成半径均匀采样数据
    X_uni, Y_uni, Zp_uni = generate_data_uniform_radius(
        num_samples=10000,
        precision=PRECISION
    )
    

#======= 测试的时候将不同数据集分开来测试 =============

    print("保存双曲 R=10 数据...")
    torch.save(X_hyp10, './datasets/test/Z_hyperbolic_R10.pt')
    torch.save(Y_hyp10, './datasets/test/A_hyperbolic_R10.pt')
    torch.save(Zp_hyp10, './datasets/test/Zp_hyperbolic_R10.pt')
    
    # 保存双曲 R=2 数据
    print("保存双曲 R=2 数据...")
    torch.save(X_hyp2, './datasets/test/Z_hyperbolic_R2.pt')
    torch.save(Y_hyp2, './datasets/test/A_hyperbolic_R2.pt')
    torch.save(Zp_hyp2, './datasets/test/Zp_hyperbolic_R2.pt')
    
    # 保存欧几里得数据
    print("保存欧几里得数据...")
    torch.save(X_euc, './datasets/test/Z_euclidean.pt')
    torch.save(Y_euc, './datasets/test/A_euclidean.pt')
    torch.save(Zp_euc, './datasets/test/Zp_euclidean.pt')
    
    # 保存弦中点数据
    print("保存弦中点数据...")
    torch.save(X_chord, './datasets/test/Z_midchord.pt')
    torch.save(Y_chord, './datasets/test/A_midchord.pt')
    torch.save(Zp_chord, './datasets/test/Zp_midchord.pt')

    # 保存半径均匀数据
    print("保存半径均匀数据...")
    torch.save(X_uni, './datasets/test/Z_uniform_radius.pt')
    torch.save(Y_uni, './datasets/test/A_uniform_radius.pt')
    torch.save(Zp_uni, './datasets/test/Zp_uniform_radius.pt')

    print("测试数据已保存！")
    
#===== 生成训练数据时将不同取样方法混合与打乱==================

    # X_all = torch.cat([X_hyp2, X_euc, X_chord, X_uni], dim=0)
    # Y_all = torch.cat([Y_hyp2, Y_euc, Y_chord, Y_uni], dim=0)
    # Zp_all = torch.cat([Zp_hyp2, Zp_euc, Zp_chord, Zp_uni], dim=0)
    
    # 6. 打乱数据
    # print("打乱数据...")
    # indices = torch.randperm(X_all.shape[0])
    # X_all = X_all[indices]
    # Y_all = Y_all[indices]
    # Zp_all = Zp_all[indices]
    
    # # 7. 保存数据
    # print("\n保存数据...")
    # torch.save(X_all, './datasets/train/v3/Z_unified_dataset.pt')
    # torch.save(Y_all, './datasets/train/v3/A_unified_dataset.pt')
    # torch.save(Zp_all, './datasets/train/v3/Zp_unified_dataset.pt')
    
    # print(f"\n{'='*60}")
    # print("数据生成完成！")
    # print(f"{'='*60}")
    # print(f"总样本数: {X_all.shape[0]}")
    # print(f"  - 双曲 R=2:  250,000")
    # print(f"  - 欧几里得:  250,000")
    # print(f"  - 弦中点:    250,000")
    # print(f"  - 半径均匀:  250,000")
    # print(f"\nZ shape: {X_all.shape}")
    # print(f"A shape: {Y_all.shape}")
    # print(f"Zp shape: {Zp_all.shape}")
    
    # # 8. 验证样本
    # print(f"\n{'='*60}")
    # print("验证随机样本:")
    # print(f"{'='*60}")
    # for i in np.random.choice(len(X_all), 3, replace=False):
    #     verify_sample(X_all, Y_all, Zp_all, i)
    
    # print("数据已保存为:")
    # print("- Z_unified_dataset.pt (输入复数)")
    # print("- A_unified_dataset.pt (变换矩阵)")
    # print("- Zp_unified_dataset.pt (输出复数)")