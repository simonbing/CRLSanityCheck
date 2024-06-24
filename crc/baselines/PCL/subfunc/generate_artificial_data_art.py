"""Data generation"""

import numpy as np
from subfunc.showdata import *


# =============================================================
# =============================================================
def generate_artificial_data(num_comp,
                             num_data,
                             ar_coef,
                             ar_order,
                             num_layer,
                             num_data_test=None,
                             random_seed=0):
    """Generate artificial data.
    Args:
        num_comp: number of components
        num_data: number of data points
        ar_coef: AR(p) coefficients of components [num_comp]
        ar_order, p of AR(p)
        num_layer: number of layers of mixing-MLP
        num_data_test: (option) number of data points (testing data, if necessary)
        random_seed: (option) random seed
    Returns:
        x: observed signals [num_comp, num_data]
        s: source signals [num_comp, num_data]
        y: labels [num_data]
        x_te: observed signals (test data) [num_comp, num_data]
        s_te: source signals (test data) [num_comp, num_data]
        y_te: labels (test data) [num_data]
        mlplayer: parameters of MLP
    """

    assert num_comp == 4
    assert ar_order == 1  # This implementation is only for ar_order=1

    # Generate source signal
    s, y = gen_source_pcl(num_comp, num_data, ar_coef, random_seed=random_seed)

    v = np.arange(num_data)
    s = np.zeros([num_data, num_comp])
    s[:, 0] = np.sin(v / 20)  # sinusoid
    s[:, 1] = (np.remainder(v, 46) - 23 < 0) * ((np.remainder(v, 23) - 11) / 9)**5 + \
              (np.remainder(v, 46) - 23 >= 0) * ((np.remainder(-v - 1, 23) - 11) / 9)**5  # funny curve (flip)
    s[:, 2] = (np.remainder(v, 54) - 27 < 0) * ((np.remainder(v, 27) - 13) / 9) + \
              (np.remainder(v, 54) - 27 >= 0) * ((np.remainder(-v - 1, 27) - 13) / 9)  # saw-tooth (flip)
    s[:, 3] = np.sign(np.random.randn(num_data)) * np.abs(np.random.randn(num_data))**1.5

    # normalize and add noise
    s = (s - np.mean(s, axis=0, keepdims=True)) / np.std(s, axis=0, keepdims=True)
    innov = np.random.laplace(0, 0.1, [num_data, num_comp])
    s = s + innov

    # ar signal
    innov = np.random.laplace(0, 1, [num_data])
    s_ar = np.zeros(num_data)
    for i in range(1, num_data):
        s_ar[i] = ar_coef[0] * s_ar[i - 1] + innov[i]
    s_ar = (s_ar - np.mean(s_ar)) / np.std(s_ar)
    s[:, 3] = s_ar

    # normalize
    s = (s - np.mean(s, axis=0, keepdims=True)) / np.std(s, axis=0, keepdims=True)

    # Apply mixing MLP
    x, mixlayer = apply_mlp_to_source(s, num_layer, random_seed=random_seed)

    # Add test data (not for demo data)
    s_te = None
    x_te = None
    y_te = None

    return x, s, y, x_te, s_te, y_te


# =============================================================
# =============================================================
def gen_source_pcl(num_comp,
                   num_data,
                   ar_coef,
                   innovation_type='laplace',
                   random_seed=0):
    """Generate source signal for PCL.
    Args:
        num_comp: number of components
        num_data: number of data points
        ar_coef: AR(1) coefficients of components [num_comp]
        innovation_type: (option) Distribution type of the innovation of source signal
        random_seed: (option) random seed
    Returns:
        source: source signals. 2D ndarray [num_comp, num_data]
        label: labels. 1D ndarray [num_data]
    """

    print("Generating source...")

    # Initialize random generator
    np.random.seed(random_seed)

    # Generate innovations
    innov = np.zeros([num_data, num_comp])
    noise_sd = np.zeros([num_comp])
    for i in range(num_comp):
        if innovation_type == 'laplace':
            noise_sd[i] = np.sqrt(1 - ar_coef[i]**2)  # var of AR signal to be 1
            innov[:, i] = np.random.laplace(0, noise_sd[i] / np.sqrt(2), [num_data])
        else:
            raise ValueError

    # Generate source signal
    s = np.zeros([num_data, num_comp])
    for i in range(1, num_data):
        s[i, :] = ar_coef * s[i - 1, :] + innov[i, :]

    # Normalize to zero-mean and unit-std
    s = (s - np.mean(s, axis=0, keepdims=True)) / np.std(s, axis=0, keepdims=True)

    y = np.ones(num_data)

    return s, y


# =============================================================
# =============================================================
def apply_mlp_to_source(s,
                        num_layer,
                        iter4condthresh=10000,
                        cond_thresh_ratio=0.25,
                        layer_name_base='ip',
                        Arange=None,
                        nonlinear_type='ReLU',
                        negative_slope=0.2,
                        random_seed=0):
    """Generate MLP and Apply it to source signal.
    Args:
        s: source signals. 2D ndarray [num_comp, num_data]
        num_layer: number of layers
        iter4condthresh: (option) number of random iteration to decide the threshold of condition number of mixing matrices
        cond_thresh_ratio: (option) percentile of condition number to decide its threshold
        layer_name_base: (option) layer name
        Arange: (option) range of value of mixing matrices
        nonlinear_type: (option) type of nonlinearity
        negative_slope: (option) parameter of leaky-ReLU
        random_seed: (option) random seed
    Returns:
        x: sensor signals. 2D ndarray [num_comp, num_data]
        mlp: parameters of mixing layers
    """

    print("Generating sensor signal...")

    if Arange is None:
        Arange = [-1, 1]

    # Subfuction to normalize mixing matrix
    def l2normalize(Amat, axis=0):
        # axis: 0=column-normalization, 1=row-normalization
        l2norm = np.sqrt(np.sum(Amat*Amat,axis))
        Amat = Amat / l2norm
        return Amat

    # Initialize random generator
    np.random.seed(random_seed)

    num_data, num_comp = s.shape

    # Determine condThresh ------------------------------------
    cond_list = np.zeros([iter4condthresh])
    for i in range(iter4condthresh):
        A = np.random.uniform(Arange[0], Arange[1], [num_comp, num_comp])
        A = l2normalize(A, axis=0)
        cond_list[i] = np.linalg.cond(A)

    cond_list.sort()  # Ascending order
    cond_thresh = cond_list[int(iter4condthresh * cond_thresh_ratio)]
    print("    cond thresh: {0:f}".format(cond_thresh))

    # Generate mixed signal -----------------------------------
    x = s.copy()
    mlp = []
    # for ln in range(num_layer - 1, -1, -1):
    for ln in range(num_layer):

        # Generate mixing matrix ------------------------------
        condA = cond_thresh + 1
        while condA > cond_thresh:
            A = np.random.uniform(Arange[0], Arange[1], [num_comp, num_comp])
            A = l2normalize(A)  # Normalize (column)
            condA = np.linalg.cond(A)
            print("    L{0:d}: cond={1:f}".format(ln, condA))
        # Bias
        b = np.zeros([1, num_comp])

        # Apply bias and mixing matrix ------------------------
        x = np.dot(x, A.T)
        x = x + b

        # Apply nonlinearity ----------------------------------
        if ln < num_layer - 1:  # No nolinearity for the first layer (source signal)
            if nonlinear_type == "ReLU":  # Leaky-ReLU
                x[x < 0] = negative_slope * x[x < 0]
            else:
                raise ValueError

        # Storege ---------------------------------------------
        mlp.append({"name": layer_name_base + str(ln), "A": A.copy(), "b": b.copy()})

    return x, mlp

