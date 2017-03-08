import matplotlib.pyplot as plt
import numpy as np
import os.path
import sys

import isomap as iso
import pca


def main():
    # Parse command line arguments.
    data_load_file = sys.argv[1]
    pca_save_name = sys.argv[2]
    iso_save_name = sys.argv[3]
    nearest_neighbors = int(sys.argv[4])
    pca_data_file = pca_save_name + ".txt"
    iso_data_file = iso_save_name + "_" + str(nearest_neighbors) + ".txt"
    # Load in the specified data.
    data = np.loadtxt(data_load_file)
    # Split out the indexes which indicate coloring.
    A, colors = np.split(data, [3], axis=1)
    colors = np.concatenate(colors)
    # Compute and plot PCA results.
    x_pca, y_pca = pca.reduce_2D(A)
    np.savetxt(pca_data_file, (x_pca, y_pca))
    plt.scatter(x_pca, y_pca, c=colors, marker="o", alpha=0.75)
    plt.title("Dimensional Reduction to 2D Using PCA")
    plt.xlabel("$e_1$")
    plt.ylabel("$e_2$")
    plt.savefig(pca_save_name + ".pdf")
    plt.close()
    # Compute and plot isomap results.
    x_iso, y_iso = iso.reduce_2D(A, nearest_neighbors)
    np.savetxt(iso_data_file, (x_iso, y_iso))
    plt.scatter(x_iso, y_iso, c=colors, marker="o", alpha=0.75)
    plt.title("Dimensional Reduction to 2D Using Isomap with %d-NN" %
              nearest_neighbors)
    plt.xlabel("$e_1$")
    plt.ylabel("$e_2$")
    plt.savefig(iso_save_name + "_" + str(nearest_neighbors) + ".pdf")
    plt.close()
    return


if __name__ == "__main__":
    main()
