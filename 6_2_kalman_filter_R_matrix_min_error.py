import matplotlib.pyplot as plt
import numpy as np
import utils

dt = 5 # Taxa de amostragem (s)

lat_vals_gt, lon_vals_gt = utils.extract_lats_lons("groundtruth.csv")
x_vals_gt, y_vals_gt = utils.coords_to_relative_cartesian(lat_vals_gt, lon_vals_gt)
x_vals_gt_uni, y_vals_gt_uni = utils.uniformize_curve(x_vals_gt, y_vals_gt, dt)

n = len(x_vals_gt_uni) # Número de amostras

# Eixo x do gráfico
t_vals = np.linspace(0, 1, n)

noise_lvl = 5

x_noisy = x_vals_gt_uni + np.random.randn(n) * noise_lvl
y_noisy = y_vals_gt_uni + np.random.randn(n) * noise_lvl

# modelo matemático
A = np.array([
    [1, 0, dt,  0, 0.5 * dt**2, 0],
    [0, 1,  0, dt, 0, 0.5 * dt**2],
    [0, 0,  1,  0, dt, 0],
    [0, 0,  0,  1, 0, dt],
    [0, 0,  0,  0, 1, 0],
    [0, 0,  0,  0, 0, 1],
])

# erro da medição
q = noise_lvl
Q = np.array([
    [1, 0],
    [0, 1]
]) * q

# matriz "máscara", que mapeia o formato (4x1) estado para o formato medição (2x1)
C = np.array([
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
])

best_mae_x = float('inf')
best_mae_y = float('inf')
best_R_x = None
best_R_y = None

q_vals = np.arange(1, 20, 0.01, dtype=float)
mae_x_vals = []
mae_y_vals = []

for q in q_vals:
    x_kalman = []
    y_kalman = []

    Sigma = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ])

    X = np.array([
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
    ])

    r = pow(10, q*(-1))
    R = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ]) * r

    print(f"testando r: 10^-{q:.4f}")

    # Loop principal do filtro de Kalman
    for i in range(n):
        # Propagação
        X = A @ X
        Sigma = A @ Sigma @ A.T + R

        # Assimilação
        z = np.array([
            [x_noisy[i]],
            [y_noisy[i]],
        ])

        K = Sigma @ C.T @ np.linalg.inv(C @ Sigma @ C.T + Q)
        X = X + K @ (z - C @ X)
        Sigma = (np.eye(6) - K @ C) @ Sigma

        x_kalman.append(X[0,0])
        y_kalman.append(X[1,0])

    x_kalman = np.array(x_kalman)
    y_kalman = np.array(y_kalman)

    mae_x = np.mean(np.abs(x_kalman - x_vals_gt_uni))
    mae_y = np.mean(np.abs(y_kalman - y_vals_gt_uni))

    mae_x_vals.append(mae_x)
    mae_y_vals.append(mae_y)

    if mae_x < best_mae_x:
        best_mae_x = mae_x
        best_R_x = r

    if mae_y < best_mae_y:
        best_mae_y = mae_y
        best_R_y = r

print("Melhor valor de R (erro do modelo):")
print(f"- Eixo X: R = 10^-{best_R_x:.10f} com MAE = {best_mae_x:.4f} m")
print(f"- Eixo Y: R = 10^-{best_R_y:.10f} com MAE = {best_mae_y:.4f} m")

_, axes = plt.subplots(1, 2, figsize=(32, 18))
axes = axes.flatten()

axes[0].plot(q_vals, mae_x_vals, linewidth=2, color='k', label='Erro do eixo X')
axes[0].set_title("Evolução do erro médio absoluto no eixo X")
axes[0].set_xlabel("q")
axes[0].grid(True)
axes[0].legend()

axes[1].plot(q_vals, mae_y_vals, linewidth=2, color='k', label='Erro do eixo Y')
axes[1].set_title("Evolução do erro médio absoluto no eixo Y")
axes[1].set_xlabel("q")
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
plt.show()
