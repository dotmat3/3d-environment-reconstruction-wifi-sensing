from utils import load_csi
from preprocessing import preprocess_sample, fillmissing

import matplotlib.pyplot as plt
import numpy as np

import os

def plot_amplitude(csi_mat, target):
    _, ntx, nrx, num_subcarriers = csi_mat.shape

    for tx in range(ntx):
        for rx in range(nrx):
            plt.figure(figsize=(10, 8))

            input_amplitude = np.absolute(csi_mat[:, tx, rx, :])
            plt.plot(input_amplitude.transpose((1, 0)))

            plt.xlim(0, num_subcarriers-1)
            plt.xlabel("Subcarrier Index")
            plt.ylabel("Amplitude dB")
            title = f"Amplitude TX{tx + 1}-RX{rx + 1}"
            plt.title(title)

            plt.savefig(os.path.join(target, title + ".pdf"))

def plot_sanitized_amplitude(final_amplitude, target):
    _, ntx, nrx, num_subcarriers = final_amplitude.shape

    # Save sanitized amplitudes
    for tx in range(ntx):
        for rx in range(nrx):
            plt.figure(figsize=(10, 8))

            input_amplitude = np.absolute(final_amplitude[:, tx, rx, :])
            plt.plot(input_amplitude.transpose(1, 0))

            plt.xlim(0, num_subcarriers-1)
            plt.xlabel("Subcarrier Index")
            plt.ylabel("Amplitude dB")
            title = f"Amplitude TX{tx + 1}-RX{rx + 1} (Sanitized)"
            plt.title(title)

            plt.savefig(os.path.join(target, title + ".pdf"))

def plot_median_amplitude(median_final_amplitude, target):
    _, num_subcarriers = median_final_amplitude.shape

    # Save median amplitudes
    plt.figure(figsize=(10, 8))

    plt.plot(median_final_amplitude.transpose(1, 0))

    plt.xlim(0, num_subcarriers-1)
    plt.xlabel("Subcarrier Index")
    plt.ylabel("Amplitude dB")
    plt.title("Amplitude TX1-RX1 (Median)")

    plt.savefig(os.path.join(target, "Amplitude TX1-RX1 (Median).pdf"))

def plot_sanitized_median_amplitude(filtered_median_amplitude, target):
    _, num_subcarriers = filtered_median_amplitude.shape

    # Save sanitized median amplitudes
    plt.figure(figsize=(10, 8))

    plt.plot(filtered_median_amplitude.transpose(1, 0))

    plt.xlim(0, num_subcarriers-1)
    plt.xlabel("Subcarrier Index")
    plt.ylabel("Amplitude dB")
    plt.title("sanitized_amplitude")

    plt.savefig(os.path.join(target, "sanitized_amplitude.pdf"))

def plot_heatmap(fill_med, target):
    plt.figure(figsize=(10, 10))
    plt.pcolormesh(fill_med, cmap="jet", shading="gouraud", edgecolor=None)
    plt.xlabel("Subcarrier Index")
    plt.ylabel("Packet No.")

    plt.savefig(os.path.join(target, "amplitude_heatmap.pdf"))

def plot_surface(fill_med, target):
    num_packets, subcarriers = fill_med.shape

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection="3d")

    X = np.arange(0, subcarriers)
    Y = np.arange(0, num_packets)
    X, Y = np.meshgrid(X, Y)

    ax.set_zlim(0, 80)
    ax.plot_surface(X, Y, fill_med, edgecolor=None, cmap="jet")
    ax.set_xlabel("Subcarrier Index")
    ax.set_ylabel("Packet No.")
    ax.set_zlabel("Amplitude dB")
    ax.view_init(azim=-135)

    plt.savefig(os.path.join(target, "amplitude_surface.pdf"))

def generate_plots_from_csi(csi, target, num_packets, ntx, nrx, num_subcarriers):
    if not os.path.exists(target):
        os.makedirs(target)

    result = preprocess_sample(csi, num_packets=num_packets, ntx=ntx, nrx=nrx, subcarriers=num_subcarriers)

    csi_mat, final_amplitude, final_phase, median_final_amplitude, filtered_median_amplitude = result

    plot_amplitude(csi_mat, target)
    plot_sanitized_amplitude(final_amplitude, target)
    plot_median_amplitude(median_final_amplitude, target)
    plot_sanitized_median_amplitude(filtered_median_amplitude, target)

    fill_med = fillmissing(filtered_median_amplitude)

    plot_heatmap(fill_med, target)
    plot_surface(fill_med, target)


def generate_plots_from_csv(path, target, num_packets, ntx, nrx, num_subcarriers):
    generate_plots_from_csi(load_csi(path), target, num_packets, ntx, nrx, num_subcarriers)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot the result of the preprocessing of the CSI data.")
    parser.add_argument("input", type=str, help="Path of the csv file containing the CSI data")
    parser.add_argument("target", type=str, help="Directory in which save the plots")
    parser.add_argument("--num-packets", default=4000, help="Number of packets to process of the data")
    parser.add_argument("--ntx", default=1, help="Number of transmitters")
    parser.add_argument("--nrx", default=1, help="Number of receivers")
    parser.add_argument("--num-subcarriers", default=50, help="Number of subcarriers to process of the data")

    args = parser.parse_args()

    generate_plots_from_csv(args.input, args.target, num_packets=args.num_packets, ntx=args.ntx, nrx=args.nrx, num_subcarriers=args.num_subcarriers)

    # Compare fill_med with output of MATLAB
    # fill_med_matlab = np.genfromtxt(r"./3-mt-experiment-5-v1.csv - fillMed.csv", delimiter=",")

    # my_data = fill_med.round(10)
    # my_data[np.isnan(my_data)] = 0

    # matlab_data = fill_med_matlab.round(10)
    # matlab_data[np.isnan(matlab_data)] = 0

    # print(fill_med[my_data != matlab_data])
    # print(fill_med_matlab[my_data != matlab_data])