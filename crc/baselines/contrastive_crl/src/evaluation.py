import sklearn.metrics
import torch
import numpy as np

from sklearn.cross_decomposition import CCA
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression

from crc.baselines.contrastive_crl.src.mcc import mean_corr_coef, mean_corr_coef_out_of_sample

def check_recovery(dataset, model, device):
    results = dict()
    samples_for_evaluation = 5000
    model.eval()
    model.to(device)
    z_gt, _, _ = dataset.sample(samples_for_evaluation, repeat_obs_samples=False)
    z_gt = z_gt[: samples_for_evaluation]

    x_gt = dataset.f(torch.tensor(z_gt, dtype=torch.float)).to(device)
    # x_int = torch.tensor(dataset.int_f, dtype=torch.float, device=device)[:samples_for_evaluation]
    z_pred = model.get_z(x_gt).cpu().detach().numpy()
    x_gt = x_gt.reshape(x_gt.size(0), -1).cpu().detach().numpy()
    regressor = LinearRegression().fit(z_pred, z_gt)
    results['R2_Z'] = regressor.score(z_pred, z_gt)

    if x_gt.size / x_gt.shape[0] < 1000:
        regressor = LinearRegression().fit(x_gt, z_gt)
        results['R2_IN'] = regressor.score(x_gt, z_gt)
    z_pred_sign_matched = z_pred * np.sign(z_pred)[:, 0:1] * np.sign(z_gt)[:, 0:1]
    if not np.isnan(z_pred).any():
        mccs = compute_mccs(z_gt, z_pred)
        mccs_sign_matched = compute_mccs(z_gt, z_pred_sign_matched)
        mccs_abs = compute_mccs(np.abs(z_gt), np.abs(z_pred))
        for k in mccs:
            results[k] = mccs[k]
            results[k +'_sign_matched'] = mccs_sign_matched[k]
            results[k +'_abs'] = mccs_abs[k]
    else:
        print('Model predicted NANs, skipping mccs evaluation')

    model.to(device)
    model.train()
    return results


def get_R2_values(x_obs, pred_obs):
    pred_obs = pred_obs - torch.mean(pred_obs, dim=0, keepdim=True)
    x_obs = x_obs - torch.mean(x_obs, dim=0, keepdim=True)
    scales = torch.sum(x_obs * pred_obs, dim=0, keepdim=True) / torch.sum(pred_obs * pred_obs, dim=0, keepdim=True)
    return 1 - torch.mean((x_obs - pred_obs * scales) ** 2, dim=0) / torch.mean(x_obs ** 2, dim=0)

def compute_mccs(x, y):
    cutoff = len(x) // 2
    ii, iinot = np.arange(cutoff), np.arange(cutoff, 2 * cutoff)
    mcc_s_in = mean_corr_coef(x=x[ii], y=y[ii])
    mcc_s_out = mean_corr_coef_out_of_sample(x=x[ii], y=y[ii], x_test=x[iinot], y_test=y[iinot])

    d = x.shape[1]
    cca_dim = min(5, d)
    cca = CCA(n_components=cca_dim, max_iter=5000)
    cca.fit(x[ii], y[ii])
    res_out = cca.transform(x[iinot], y[iinot])
    mcc_w_out = mean_corr_coef(res_out[0], res_out[1])
    res_in = cca.transform(x[ii], y[ii])
    mcc_w_in = mean_corr_coef(res_in[0], res_in[1])
    return {"mcc_s_in": mcc_s_in, "mcc_s_out": mcc_s_out, "mcc_w_in": mcc_w_in, "mcc_w_out": mcc_w_out}


def print_cor_coef(x_obs, pred_obs):
    pred_obs = pred_obs - torch.mean(pred_obs, dim=0, keepdim=True)
    x_obs = x_obs - torch.mean(x_obs, dim=0, keepdim=True)
    pred_obs_copy = pred_obs.detach().clone()
    x_obs_copy = x_obs.detach().clone()
    pred_obs_copy = pred_obs_copy.unsqueeze(1)
    x_obs_copy = x_obs_copy.unsqueeze(2)
    cors = torch.mean(pred_obs_copy * x_obs_copy, dim=0)
    var_x = torch.std(x_obs_copy, dim=0)
    var_pred = torch.std(pred_obs_copy, dim=0)
    print((1 / var_pred).view(-1, 1) * cors * (1/var_x).view(1, -1))


def evaluate_graph_metrics(W_true, W, thresh=.3, nr_edges=1):
    B_true = np.where(np.abs(W_true)>.01, 1, 0)
    np.fill_diagonal(W, 0.)
    B = np.where(np.abs(W)>thresh, 1, 0)
    np.fill_diagonal(B, 0)
    loss_dict = evaluate_graph_metrics_for_thresholded_graphs(B_true, B)
    opt_threshold = get_opt_thresh(B_true, W)
    B = np.where(np.abs(W)>opt_threshold, 1, 0)
    np.fill_diagonal(B, 0)
    loss_dict_opt = evaluate_graph_metrics_for_thresholded_graphs(B_true, B)
    loss_dict['SHD_opt'] = loss_dict_opt['SHD']
    loss_dict['FPR_opt'] = loss_dict_opt['FPR']
    loss_dict['FDR_opt'] = loss_dict_opt['FDR']
    loss_dict['TPR_opt'] = loss_dict_opt['TPR']
    loss_dict['opt_thresh'] = opt_threshold

    edge_threshold = get_edge_threshold(W, nr_edges)
    B = np.where(np.abs(W)>edge_threshold, 1, 0)
    np.fill_diagonal(B, 0)
    loss_dict_opt = evaluate_graph_metrics_for_thresholded_graphs(B_true, B)
    loss_dict['SHD_edge_matched'] = loss_dict_opt['SHD']
    loss_dict['FPR_edge_matched'] = loss_dict_opt['FPR']
    loss_dict['FDR_edge_matched'] = loss_dict_opt['FDR']
    loss_dict['TPR_edge_matched'] = loss_dict_opt['TPR']
    loss_dict['edge_thresh'] = opt_threshold
    loss_dict['auroc'] = get_auroc(B_true, W)
    return loss_dict


def get_edge_threshold(W, nr_edges):
    W_abs = np.abs(W).reshape(-1)
    return W_abs[np.argsort(W_abs)[-(nr_edges+1)]]

def get_auroc(B_true, W):
    W_abs = np.abs(W)
    B_true_del = B_true[~np.eye(B_true.shape[0], dtype=bool)].reshape(B_true.shape[0], -1)
    W_abs_del = W_abs[~np.eye(W_abs.shape[0], dtype=bool)].reshape(W_abs.shape[0], -1)
    if np.sum(B_true_del) < 1:
        return -1
    return roc_auc_score(B_true_del.reshape(-1), W_abs_del.reshape(-1))


def evaluate_graph_metrics_for_thresholded_graphs(B_true, B):
    loss_dict = dict()
    loss_dict['SHD'] = np.sum(np.abs(B_true - B))
    if np.sum(B_true) < 1:
        loss_dict['TPR'] = 1
    else:
        loss_dict['TPR'] = np.sum(B_true * B) / np.sum(B_true)
    loss_dict['FPR'] = np.sum((1-B_true) * B) / (np.sum(1-B_true) - B_true.shape[0])
    if np.sum(B) < 1:
        loss_dict['FDR'] = 0
    else:
        loss_dict['FDR'] = np.sum((1-B_true) * B) / np.sum(B)
    return loss_dict


def get_opt_thresh(B_true, W):
    thresholds = np.arange(0., 2., .005)
    opt_thresh = 0.
    min_shd = np.inf
    for t in thresholds:
        B = np.where(np.abs(W) > t, 1, 0)
        shd_t = np.sum(np.abs(B_true - B))
        if shd_t < min_shd:
            opt_thresh = t
            min_shd = shd_t
    return opt_thresh





class LossCollection:
    def __init__(self):
        self.loss_dict = dict()
        self.steps = 0

    def add_loss(self, loss_updates, bs):
        for key, value in loss_updates.items():
            if key in self.loss_dict:
                self.loss_dict[key] = self.loss_dict[key] + bs * value
            else:
                self.loss_dict[key] = bs * value
        self.steps += bs

    def reset(self):
        self.steps = 0
        self.loss_dict = dict()

    def get_mean_loss(self):
        mean_loss_dict = dict()
        for key, value in self.loss_dict.items():
            mean_loss_dict[key] = value / self.steps
        return mean_loss_dict

    def print_mean_loss(self):
        print("Current mean losses are:")
        print(self.get_mean_loss())