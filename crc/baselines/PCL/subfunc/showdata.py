import numpy as np
import colorcet as cc
import matplotlib.pyplot as plt


# =============================================================
# showmat
# =============================================================
def showmat(x, xlabel=None, ylabel=None, fontsize=14, crange=None, figsize=None, xticklabel=None, yticklabel=None):
    """Show 2D ndarray as matrix.
    Args:
        x: 2D ndarray
        xlabel: (option) x-axis label
        ylabel: (option) y-axis label
        fontsize: (option) font size
        crange: (option) colormap range, [min, max] or "maxabs"
        figsize: (option) figure size
        xticklabel: (option) xticklabel
        yticklabel: (option) yticklabel
    """

    # Prepare plot data ---------------------------------------
    if figsize is None:
        figsize = [1, 1]
    x = x.copy()
    if len(x.shape) > 2:
        print('X has to be matrix or vector')
        return

    if x.shape[0] == 1 or x.shape[1] == 1:
        x = x.reshape(np.sqrt(x.size), np.sqrt(x.size))

    # Plot ----------------------------------------------------
    plt.figure(figsize=(8*figsize[0], 6*figsize[1]))

    plt.imshow(x, interpolation='none', aspect='auto')
    plt.colorbar()

    # Color range
    if not(crange is None):
        if len(crange) == 2:
            plt.clim(crange[0], crange[1])

        elif crange == 'maxabs':
            xmaxabs = np.absolute(x).max()
            plt.clim(-xmaxabs, xmaxabs)

    if not(xlabel is None):
        plt.xlabel(xlabel)
    if not(ylabel is None):
        plt.ylabel(ylabel)
    if xticklabel is not None:
        plt.gca().set_xticks(np.arange(0, x.shape[1]))
        plt.gca().set_xticklabels(xticklabel)
    if yticklabel is not None:
        plt.gca().set_yticks(np.arange(0, x.shape[0]))
        plt.gca().set_yticklabels(yticklabel)
        # ax.set_xticklabels(xticklabel)

    plt.rcParams['font.size'] = fontsize

    plt.ion()
    plt.show()
    plt.pause(0.001)


# =============================================================
# showtimedata
# =============================================================
def showtimedata(x, xlabel='Time', ylabel='Channel', fontsize=14, linewidth=1.5,
                 intervalstd=10, figsize=None, linecolor=None):
    """Show 2D ndarray as time series
    Args:
        x: signals. 2D ndarray [num_channel, num_time]
        xlabel: (option) x-axis label
        ylabel: (option) y-axis label
        fontsize: (option) font size
        linewidth: (option) width of lines
        intervalstd: (option) interval between lines based on maximum std.
        figsize: (option) figure size
    """

    # Prepare plot data ---------------------------------------
    if figsize is None:
        figsize = [2, 1]
    x = x.copy()
    if len(x.shape) > 2:
        print('X has to be matrix or vector')
        return

    if x.shape[0] == 1:
        x = x.reshape([-1, 1])

    num_data = x.shape[0]
    num_dim = x.shape[1]

    # normalize the signal
    x = x - np.mean(x, axis=0, keepdims=True)
    x = x / np.max(np.abs(x), axis=0, keepdims=True) * 0.45
    # x = (x - np.mean(x, axis=0, keepdims=True)) / np.std(x, axis=0, keepdims=True)

    vinterval = 1
    # vinterval = x.std(axis=0).max() * intervalstd
    vpos = vinterval * (np.arange(num_dim, 0, -1) - 1)
    x = x + vpos[None, :]

    # Plot ----------------------------------------------------
    plt.figure(figsize=(8*figsize[0], 6*figsize[1]))
    if linecolor is None:
        if num_dim < 5:
            cmap = [[213/256, 94/256, 0], [230/256, 159/256, 0], [0, 158/256, 115/256], [0, 114/256, 178/256]]
        else:
            cmap = cc.glasbey
    else:
        cmap = linecolor
    # plt.plot(x, linewidth=linewidth, color=cmap)

    for i in range(num_dim):
        plt.plot([0, num_data - 1], [vpos[i], vpos[i]], linewidth=1, color=[0.5, 0.5, 0.5])
        plt.plot(list(range(num_data)), x[:, i], linewidth=linewidth, color=cmap[i])

    plt.xlim(0, num_data - 1)
    plt.ylim(x.min(), x.max())

    ylabels = [str(num) for num in range(num_dim)]
    plt.yticks(vpos.reshape(-1), ylabels)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.rcParams['font.size'] = fontsize

    plt.ion()
    plt.show()
    plt.pause(0.001)


# =============================================================
# showhist
# =============================================================
def showhist(x, bins=100, xlabel=None, ylabel=None, fontsize=14, figsize=None):
    """Show 2D ndarray as matrix.
    Args:
        x: 2D ndarray
        bins: number of bins
        xlabel: (option) x-axis label
        ylabel: (option) y-axis label
        fontsize: (option) font size
        figsize: (option) figure size
    """

    # Prepare plot data ---------------------------------------
    if figsize is None:
        figsize = [1, 1]
    x = x.copy()
    if len(x.shape) > 2:
        print('X has to be matrix or vector')
        return

    plt.figure(figsize=(8*figsize[0], 6*figsize[1]))
    plt.hist(x, bins=bins)

    if not(xlabel is None):
        plt.xlabel(xlabel)
    if not(ylabel is None):
        plt.ylabel(ylabel)

    plt.rcParams['font.size'] = fontsize
