import matplotlib.pyplot as plt
import numpy as np
import csv
from pyproj import Transformer, CRS
from datetime import datetime
from scipy.interpolate import interp1d

# create_array creates values for plotting graphs.
#
# Example:
# 
# >>> x = create_array(0, 0.2, 100)
# >>> x
# [0, 0.2, 0.4, 0.6, ..., 99.6, 99.8, 100.0]
# 
# (this was made before I learned about numpy's linspace)
# 
def create_array(start, step, end):
    arr = []
    i = 0
    while (i*step)+start <= end:
        arr.append(round(i*step + start, 3))
        i = i+1
    return arr

def plot_avg_filters_const_sig(x, y_noisy_sig, y_bulk_avg, y_iter_avg):
    plt.plot(x, y_noisy_sig, linewidth=0.7, label='sinal ruidoso')
    plt.plot(x, y_bulk_avg, linewidth=2, color='r', label='média comum')
    plt.plot(x, y_iter_avg, linewidth=2, color='g', label='média iterativa')
    plt.annotate('%0.4f' % y_bulk_avg[-1], xy=(1, y_bulk_avg[-1]), xytext=(6, 5), xycoords=('axes fraction', 'data'),  color='r', fontsize=12, textcoords='offset points')
    plt.annotate('%0.4f' % y_iter_avg[-1], xy=(1, y_iter_avg[-1]), xytext=(6, -10), xycoords=('axes fraction', 'data'),  color='g', fontsize=12, textcoords='offset points')
    plt.scatter(x[-1], y_bulk_avg[-1], color='r')
    plt.scatter(x[-1], y_iter_avg[-1], color='g')
    plt.legend(loc="upper left")
    plt.title("Atuação de filtros de média comum e iterativo sobre sinal ruidoso")
    plt.ylabel("Altura (m)")
    plt.xlabel("tempo (s)")
    plt.ylim((1.5, 2.5))
    plt.subplots_adjust(top=0.93, bottom=0.11, left=0.075, right=0.94, hspace=0.2, wspace=0.2)
    plt.show()

def plot_avg_filters_sine_sig(x, y_groundtruth, y_noisy_sig, y_bulk_avg, y_iter_avg):
    plt.plot(x, y_groundtruth, linewidth=2, color='k', label='sinal real')
    plt.plot(x, y_noisy_sig, linewidth=0.7, label='sinal ruidoso')
    plt.plot(x, y_bulk_avg, linewidth=2, color='r', label='média comum')
    plt.plot(x, y_iter_avg, linewidth=2, color='g', label='média iterativa')
    plt.legend(loc="upper left")
    plt.title("Atuação de filtros de média comum e iterativo sobre sinal dinâmico ruidoso")
    plt.ylabel("Altura (m)")
    plt.xlabel("tempo (s)")
    plt.ylim((1.5, 2.5))
    plt.subplots_adjust(top=0.93,bottom=0.11,left=0.075,right=0.94,hspace=0.2,wspace=0.2)
    plt.show()

def plot_moving_avg_filters_sine_sig(x, y_groundtruth, y_noisy_sig, y_bulk_mov_avg, y_iter_mov_avg, w):
    plt.plot(x, y_groundtruth, linewidth=1.5, color='k', label='sinal real')
    plt.plot(x, y_noisy_sig, linewidth=0.7, label='sinal ruidoso')
    # plt.plot(x, y_bulk_mov_avg, linewidth=2, color='r', label='média móvel comum')    # Comentado por não adicionar valor
                                                                                        # visual à demonstração do filtro,
                                                                                        # em comparação com sua versão
                                                                                        # iterativa.
    plt.plot(x, y_iter_mov_avg, linewidth=1.7, color='g', label='média móvel iterativa')
    plt.legend(loc="upper left")
    plt.title(f"Atuação do filtro de média móvel iterativo sobre sinal dinâmico ruidoso - Janela de tamanho {w}")
    plt.ylabel("Altura (m)")
    plt.xlabel("tempo (s)")
    plt.ylim((1.5, 2.5))
    plt.subplots_adjust(top=0.93,bottom=0.11,left=0.075,right=0.94,hspace=0.2,wspace=0.2)
    plt.show()

def plot_exp_moving_avg_filter_sine_sig(x, y_groundtruth, y_noisy_sig, y_exp_mov_avg, alpha):
    plt.plot(x, y_groundtruth, linewidth=1.5, color='k', label='sinal real')
    plt.plot(x, y_noisy_sig, linewidth=0.7, label='sinal ruidoso')
    plt.plot(x, y_exp_mov_avg, linewidth=1.7, color='g', label='média móvel exponencial')
    plt.legend(loc="upper left")
    plt.title(f"Atuação do filtro de média móvel exponencial sobre sinal dinâmico ruidoso - Alfa de {alpha}")
    plt.ylabel("Altura (m)")
    plt.xlabel("tempo (s)")
    plt.ylim((1.5, 2.5))
    plt.subplots_adjust(top=0.93,bottom=0.11,left=0.075,right=0.94,hspace=0.2,wspace=0.2)
    plt.show()

def plot_kalman_filter_1D_position_pos(t, y_groundtruth_pos, y_kalman_pos, y_noisy_pos):
    plt.plot(t, y_noisy_pos, linewidth=.8, label='sinal ruidoso')
    plt.plot(t, y_groundtruth_pos, linewidth=4, color='k', label='posição real')
    plt.plot(t, y_kalman_pos, linewidth=1.7, color='limegreen', label='estimativa do filtro de Kalman')
    plt.ylabel("Posição (m)")
    plt.xlabel("tempo (s)")
    plt.legend()
    plt.show()

def plot_kalman_filter_1D_position_vel(t, y_groundtruth_vel, y_kalman_vel):
    plt.plot(t, y_groundtruth_vel, linewidth=1.7, color='k', label='velocidade real')
    plt.plot(t, y_kalman_vel, linewidth=3, color='g', label='estimativa do filtro de Kalman')
    plt.ylabel("Velocidade (m/s)")
    plt.xlabel("tempo (s)")
    plt.legend()
    plt.show()

def plot_kalman_filter_1D_position_acc(t, y_groundtruth_acc, y_kalman_acc):
    plt.plot(t, y_groundtruth_acc, linewidth=1.7, color='k', label='aceleração real')
    plt.plot(t, y_kalman_acc, linewidth=3, color='g', label='estimativa do filtro de Kalman')
    plt.ylabel("Aceleração (m/s²)")
    plt.xlabel("tempo (s)")
    plt.legend()
    plt.show()

def extract_lats_lons(file_name: str) -> tuple[list, list]:
    lat_vals = []
    lon_vals = []

    with open(file_name) as file:
        reader = csv.DictReader(file, delimiter="\t")
        for row in reader:
            lat_vals.insert(0, float(row["latitude"]))
            lon_vals.insert(0, float(row["longitude"]))
    
    return lat_vals, lon_vals

def coords_to_relative_cartesian(lat_vals, lon_vals) -> tuple[list, list]:
    first_lat, first_lon = lat_vals[0], lon_vals[0]

    utm_crs = CRS.from_proj4(f"+proj=utm +zone={(int((first_lon + 180) / 6) + 1)} +south +ellps=WGS84 +units=m +no_defs")

    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_crs.to_epsg()}", always_xy=True)

    x0, y0 = transformer.transform(first_lon, first_lat)
    x_vals = []
    y_vals = []

    # Converter todos os pontos para (x, y) relativos à origem.
    for i in range(len(lat_vals)):
        x, y = transformer.transform(lon_vals[i], lat_vals[i])

        x_vals.append(x - x0)
        y_vals.append(y - y0)
    
    return x_vals, y_vals

def uniformize_curve(x_vals, y_vals, spacing):
    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)

    dx = np.diff(x_vals)
    dy = np.diff(y_vals)
    segment_lengths = np.sqrt(dx**2 + dy**2)

    cumulative_length = np.insert(np.cumsum(segment_lengths), 0, 0)

    interp_x = interp1d(cumulative_length, x_vals)
    interp_y = interp1d(cumulative_length, y_vals)

    total_length = cumulative_length[-1]
    new_lengths = np.arange(0, total_length, spacing)

    new_x = interp_x(new_lengths)
    new_y = interp_y(new_lengths)

    return new_x, new_y

def test_errors(x_gt, y_gt, x_noisy, y_noisy, x_filtered, y_filtered):
    print("Erros do caminho ruidoso:")
    mae_noisy_x = np.mean(np.abs(x_noisy - x_gt))
    mse_noisy_x = np.mean((x_noisy - x_gt)**2)
    rmse_noisy_x = np.sqrt(mse_noisy_x)
    distance_noisy_x = np.sum(np.abs(x_noisy - x_gt))
    print("Eixo X:")
    print(f"- MAE (Erro Médio Absoluto): {mae_noisy_x:.4f} m")
    print(f"- RMSE (Raiz do Erro Quadrático Médio): {rmse_noisy_x:.4f} m")
    print(f"- Distância acumulada entre os pontos: {distance_noisy_x:.4f} m")

    mae_noisy_y = np.mean(np.abs(y_noisy - y_gt))
    mse_noisy_y = np.mean((y_noisy - y_gt)**2)
    rmse_noisy_y = np.sqrt(mse_noisy_y)
    distance_noisy_y = np.sum(np.abs(y_noisy - y_gt))
    print("Eixo Y:")
    print(f"- MAE (Erro Médio Absoluto): {mae_noisy_y:.4f} m")
    print(f"- RMSE (Raiz do Erro Quadrático Médio): {rmse_noisy_y:.4f} m")
    print(f"- Distância acumulada entre os pontos: {distance_noisy_y:.4f} m")

    print()
    print("Erros com o filtro de Kalman")
    mae_x = np.mean(np.abs(x_filtered - x_gt))
    mse_x = np.mean((x_filtered - x_gt)**2)
    rmse_x = np.sqrt(mse_x)
    distance_x = np.sum(np.abs(x_filtered - x_gt))
    print("Eixo X")
    print(f"- MAE (Erro Médio Absoluto): {mae_x:.4f} m")
    print(f"- RMSE (Raiz do Erro Quadrático Médio): {rmse_x:.4f} m")
    print(f"- Distância acumulada entre os pontos: {distance_x:.4f} m")

    mae_y = np.mean(np.abs(y_filtered - y_gt))
    mse_y = np.mean((y_filtered - y_gt)**2)
    rmse_y = np.sqrt(mse_y)
    distance_y = np.sum(np.abs(y_filtered - y_gt))
    print("Eixo Y")
    print(f"- MAE (Erro Médio Absoluto): {mae_y:.4f} m")
    print(f"- RMSE (Raiz do Erro Quadrático Médio): {rmse_y:.4f} m")
    print(f"- Distância acumulada entre os pontos: {distance_y:.4f} m")

def plot_kalman_filter_1D_position_sigmas(t, sigmas, dim):
    _, axes = plt.subplots(dim, dim, figsize=(16, 12))
    axes = axes.flatten()

    for i in range(dim):
        for j in range(dim):
            idx = i * dim + j
            label = f"Σ[{i},{j}]"
            values = [sigma[i, j] for sigma in sigmas]
            axes[idx].plot(t, values, linewidth=1.7, color='k', label=label)
            axes[idx].set_xlabel("tempo (s)")
            axes[idx].legend()
            axes[idx].grid(True)

    plt.tight_layout()
    plt.show()
