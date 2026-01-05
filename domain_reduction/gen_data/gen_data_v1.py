#随机生成一个基本域外的点，然后根据迭代算法规约到基本域内

from fractions import Fraction
import random
import torch


def random_fraction(max_abs_num=100, max_den=100):
    p = random.randint(-max_abs_num, max_abs_num)
    q = random.randint(1, max_den)
    return Fraction(p, q)

def apply_matrix(a, b, c, d, z):
    x, y = z
    # 复数除法：(a*z+b)/(c*z+d)
    # 分子
    num_re = a * x + b
    num_im = a * y
    # 分母
    den_re = c * x + d
    den_im = c * y
    denom = den_re * den_re + den_im * den_im
    z_re = (num_re * den_re + num_im * den_im) / denom
    z_im = (num_im * den_re - num_re * den_im) / denom
    return (z_re, z_im)

def reduce_to_fundamental_domain(z, max_steps=1000):
    a, b, c, d = 1, 0, 0, 1
    x, y = float(z[0]), float(z[1])
    steps = 0
    while abs(x) >= 0.5:
        if steps > max_steps:
            # print("Reduce failed: too many steps in translation.")
            return None, None
        n = int(round(x))
        b -= n * d
        a -= n * c
        x -= n
        steps += 1

    while x*x + y*y < 1:
        if steps > max_steps:
            # print("Reduce failed: too many steps in inversion.")
            return None, None
        z = (-x/(x*x + y*y), y/(x*x + y*y))
        a, b, c, d = c, d, -a, -b
        x, y = float(z[0]), float(z[1])

        while abs(x) >= 0.5:
            if steps > max_steps:
                # print("Reduce failed: too many steps in translation (after inversion).")
                return None, None
            
            n = int(round(x))
            b -= n * d
            a -= n * c
            x -= n
            steps += 1
        steps += 1
    return z, (a, b, c, d)

def random_outside_fundamental_domain(max_abs_num=100, max_den=100, tries=1000):

    for _ in range(tries):
        x = random_fraction(max_abs_num, max_den)
        y = random_fraction(max_abs_num, max_den)
        if y <= 0:
            continue
        if abs(x) >= 0.5 or x*x + y*y < 1:
            return (x, y)
        
    return None

def generate_data_outside(num_samples=100, max_abs_num=100, max_den=100):
    xs = []
    ys = []
    zs_prime = []
    count = 0
    while count < num_samples:
        z = random_outside_fundamental_domain(max_abs_num, max_den)
        z_prime, A = reduce_to_fundamental_domain(z)
        if z_prime is None:
            continue

        a, b, c, d = A
        z_prime_exact = apply_matrix(a, b, c, d, z)
        xs.append([z[0].numerator, z[0].denominator, z[1].numerator, z[1].denominator])
        ys.append(A)
        zs_prime.append([z_prime_exact[0], z_prime_exact[1]])
        count += 1
        if count % 10000 == 0:
            print(f"Generated {count} samples...")
    X_tensor = torch.tensor(xs, dtype=torch.long)
    Y_tensor = torch.tensor(ys, dtype=torch.long)
    Zp_tensor = torch.tensor(zs_prime, dtype=torch.float)
    return X_tensor, Y_tensor, Zp_tensor

if __name__ == "__main__":
    X, Y, Zp = generate_data_outside(num_samples=10000)
    torch.save(X, 'Z_t.pt')
    torch.save(Y, 'A_t.pt')
    print("Z shape:", X.shape, "A shape:", Y.shape, "Zp shape:", Zp.shape)
    for i in range(100):
        print("z =", X[i].tolist(), " --> A =", Y[i].tolist(), " --> z' =", Zp[i].tolist())