import matplotlib.pyplot as plt
import numpy as np

def plot_USA_Boundary(LW=2):
    """
    Plots the boundary of the USA using data from a file.

    Args:
        LW (int, optional): Line width for the boundary. Defaults to 2.
    """
    # Load data from 'USA.txt'
    W = np.loadtxt('USA.txt')

    # Remove Alaska and Hawaii
    kilx = np.where(W[:, 1] < -140)[0]
    kily = np.where(W[:, 2] > 50)[0]
    W = np.delete(W, np.concatenate((kilx, kily)), axis=0)

    # Adjust the 4th column
    W[:, 3] = W[:, 3] + 1

    # Plot the boundary segments
    N = len(W)
    plot_bel(1, 7, W, LW)
    plot_bel(8, 35, W, LW)
    plot_bel(36, 142, W, LW)
    plot_bel(143, 3089, W, LW)
    plot_bel(3090, 3102, W, LW)
    plot_bel(3103, 4698, W, LW)
    plot_bel(4699, 4733, W, LW)
    plot_bel(4734, 4757, W, LW)
    plot_bel(4758, 4759, W, LW)
    plot_bel(4760, 6436, W, LW)
    plot_bel(6437, 6461, W, LW)
    plot_bel(6462, 10077, W, LW)
    plot_bel(10078, 10714, W, LW)
    plot_bel(10715, N, W, LW)

    plt.axis([-130, -60, 20, 50])
    # plt.show()


def plot_bel(st, ed, W, LW):
    """
    Plots a segment of the boundary.

    Args:
        st (int): Starting index.
        ed (int): Ending index.
        W (numpy.ndarray): Data array.
        LW (int): Line width.
    """
    a = st
    for i in range(st + 1, ed + 1):
        if W[i - 1, 0] == W[i - 2, 0]:
            b = i
        else:
            plt.plot(W[a - 1:b, 1], W[a - 1:b, 2], 'k', linewidth=LW)
            a = b + 1
            b = a

    plt.plot(W[a - 1:b, 1], W[a - 1:b, 2], 'k', linewidth=LW)