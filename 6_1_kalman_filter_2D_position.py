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

# Estimativa inicial
X = np.array([
    [0],
    [0],
    [0],
    [0],
    [0],
    [0],
])

# modelo matemático
A = np.array([
    [1, 0, dt,  0, 0.5 * dt**2, 0],
    [0, 1,  0, dt, 0, 0.5 * dt**2],
    [0, 0,  1,  0, dt, 0],
    [0, 0,  0,  1, 0, dt],
    [0, 0,  0,  0, 1, 0],
    [0, 0,  0,  0, 0, 1],
])

# matriz de covariância "certeza geral do filtro"
Sigma = np.array([
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1],
])

# erro do modelo
r = 10**-7
R = np.array([
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1],
]) * r

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

x_kalman = []
y_kalman = []
Sigmas = []

# Loop principal do filtro de Kalman
for i in range(n):
    Sigmas.append(Sigma)

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

# ** ANALISANDO ERRO ** 
utils.test_errors(x_vals_gt_uni, y_vals_gt_uni, x_noisy, y_noisy, x_kalman, y_kalman)

# ** PLOT ** 
plt.plot(x_vals_gt_uni, y_vals_gt_uni, linewidth=1, color='k', label='groundtruth')
plt.plot(x_kalman, y_kalman, linewidth=2, color='limegreen', label='estimativa do filtro de Kalman')
plt.title("Trajetória estimada pelo filtro de Kalman")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.plot(x_kalman, y_kalman, linewidth=2, color='limegreen', label='estimativa do filtro de Kalman')
plt.title("Trajetória estimada pelo filtro de Kalman")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.plot(x_vals_gt_uni, y_vals_gt_uni, linewidth=2, color='k', label='groundtruth')
plt.plot(x_noisy, y_noisy, marker='.', linestyle='none', color='red', markersize=2, label='medição')
plt.title("Trajetória real, em coordenadas relativas (x, y)\ncom medições ruidosas")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.plot(x_noisy, y_noisy, marker='.', linestyle='none', linewidth=0.5, color='red', markersize=2, label='medição')
plt.title("Medições das coordenadas do trajeto com ruído simulado")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.plot(t_vals * len(t_vals), x_vals_gt_uni, linewidth=3, color='k', label='groundtruth')
plt.plot(t_vals * len(t_vals), x_kalman, linewidth=2, color='limegreen', label='estimativa do filtro de Kalman')
plt.plot(t_vals * len(t_vals), x_noisy, marker='.', linestyle='none', color='red', markersize=1, label='medição')
plt.title("Análise do erro no eixo x")
plt.xlabel("tempo (s)")
plt.ylabel("x (m)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.plot(t_vals * len(t_vals), y_vals_gt_uni, linewidth=3, color='k', label='groundtruth')
plt.plot(t_vals * len(t_vals), y_kalman, linewidth=2, color='limegreen', label='estimativa do filtro de Kalman')
plt.plot(t_vals * len(t_vals), y_noisy, marker='.', linestyle='none', color='red', markersize=1, label='medição')
plt.title("Análise do erro no eixo y")
plt.xlabel("tempo (s)")
plt.ylabel("y (m)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

utils.plot_kalman_filter_1D_position_sigmas(t_vals * len(t_vals), Sigmas, 6)