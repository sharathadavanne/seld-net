#
# Implements the core metrics from sound event detection evaluation module http://tut-arg.github.io/sed_eval/ and
# the DOA metrics explained in the SELDnet paper
#

import numpy as np
from sklearn.metrics import confusion_matrix
from IPython import embed

eps = np.finfo(np.float).eps

###############################################################
# Scoring functions
###############################################################


def reshape_3Dto2D(A):
    return A.reshape(A.shape[0] * A.shape[1], A.shape[2])


def f1_overall_framewise(O, T):
    if len(O.shape) == 3:
        O, T = reshape_3Dto2D(O), reshape_3Dto2D(T)
    TP = ((2 * T - O) == 1).sum()
    Nref, Nsys = T.sum(), O.sum()

    prec = float(TP) / float(Nsys + eps)
    recall = float(TP) / float(Nref + eps)
    f1_score = 2 * prec * recall / (prec + recall + eps)
    return f1_score


def er_overall_framewise(O, T):
    if len(O.shape) == 3:
        O, T = reshape_3Dto2D(O), reshape_3Dto2D(T)

    FP = np.logical_and(T == 0, O == 1).sum(1)
    FN = np.logical_and(T == 1, O == 0).sum(1)

    S = np.minimum(FP, FN).sum()
    D = np.maximum(0, FN-FP).sum()
    I = np.maximum(0, FP-FN).sum()

    Nref = T.sum()
    ER = (S+D+I) / (Nref + 0.0)
    return ER


def f1_framewise(O, T):
    # This is wrongly calculated f1 score where per frame F1-score is
    # caluclated and later mean of these is taken. Use this only for legacy stuff
    if len(O.shape) == 3:
        O, T = reshape_3Dto2D(O), reshape_3Dto2D(T)
    TP = ((2 * T - O) == 1).sum(axis=1)
    Nref, Nsys = T.sum(axis=1), O.sum(axis=1)

    prec = (TP + eps) / (Nsys + eps)
    recall = (TP + eps) / (Nref + eps)
    f1_score = 2 * prec * recall / (prec + recall + eps)
    return f1_score


def f1_1sec(O, T, block_size):
    if len(O.shape) == 3:
        O, T = reshape_3Dto2D(O), reshape_3Dto2D(T)
    new_size = int(O.shape[0] / block_size)
    O_block = np.zeros((new_size, O.shape[1]))
    T_block = np.zeros((new_size, O.shape[1]))
    for i in range(0, new_size):
        O_block[i,] = np.max(O[int(i * block_size):int(i * block_size + block_size - 1), ], axis=0)
        T_block[i,] = np.max(T[int(i * block_size):int(i * block_size + block_size - 1), ], axis=0)
    return f1_framewise(O_block, T_block)


def f1_overall_1sec(O, T, block_size):
    if len(O.shape) == 3:
        O, T = reshape_3Dto2D(O), reshape_3Dto2D(T)
    new_size = int(np.ceil(O.shape[0] / block_size))
    O_block = np.zeros((new_size, O.shape[1]))
    T_block = np.zeros((new_size, O.shape[1]))
    for i in range(0, new_size):
        O_block[i,] = np.max(O[int(i * block_size):int(i * block_size + block_size - 1), ], axis=0)
        T_block[i,] = np.max(T[int(i * block_size):int(i * block_size + block_size - 1), ], axis=0)
    return f1_overall_framewise(O_block, T_block)


def er_overall_1sec(O, T, block_size):
    if len(O.shape) == 3:
        O, T = reshape_3Dto2D(O), reshape_3Dto2D(T)
    new_size = int(O.shape[0] / (block_size))
    O_block = np.zeros((new_size, O.shape[1]))
    T_block = np.zeros((new_size, O.shape[1]))
    for i in range(0, new_size):
        O_block[i,] = np.max(O[int(i * block_size):int(i * block_size + block_size - 1), ], axis=0)
        T_block[i,] = np.max(T[int(i * block_size):int(i * block_size + block_size - 1), ], axis=0)
    return er_overall_framewise(O_block, T_block)


def compute_sed_scores(pred, y, nb_frames_1s):
    """Compute TUT metrics

    Parameters
    ----------
    pred : matrix
        predicted matrix / system output

    y : matrix
        reference matrix

    hop_length_seconds : float
        used frame hop length

    Returns
    -------
    scores : dict
    """
    f1o = f1_overall_1sec(pred, y, nb_frames_1s)
    ero = er_overall_1sec(pred, y, nb_frames_1s)
    scores = [ero, f1o]
    return scores


def cart2sph(x,y,z):
    azimuth = np.arctan2(y,x)
    elevation = np.arctan2(z,np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)
    return azimuth, elevation, r


def sph2cart(azimuth,elevation,r):
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    return x, y, z


def compute_doa_scores_regr_xy(pred, gt, pred_sed, gt_sed):
    nb_src_gt_list = np.zeros(gt.shape[0]).astype(int)
    nb_src_pred_list = np.zeros(gt.shape[0]).astype(int)
    good_frame_cnt = 0
    less_frame_cnt = 0
    more_frame_cnt = 0
    doa_loss_gt = 0.0
    doa_loss_gt_cnt = 0
    doa_loss_pred = 0.0
    doa_loss_pred_cnt = 0
    nb_sed = gt_sed.shape[-1]

    for frame_cnt, sed_frame in enumerate(gt_sed):
        nb_src_gt_list[frame_cnt] = int(np.sum(sed_frame))
        nb_src_pred_list[frame_cnt] = int(np.sum(pred_sed[frame_cnt]))
        if nb_src_gt_list[frame_cnt] > nb_src_pred_list[frame_cnt]:
            less_frame_cnt = less_frame_cnt + 1
        elif nb_src_gt_list[frame_cnt] < nb_src_pred_list[frame_cnt]:
            more_frame_cnt = more_frame_cnt + 1
        else:
            good_frame_cnt = good_frame_cnt + 1

        # DOA Loss with respect to groundtruth
        doa_frame_gt_x = gt[frame_cnt][:nb_sed][sed_frame == 1]
        doa_frame_gt_y = gt[frame_cnt][nb_sed:2*nb_sed][sed_frame == 1]

        doa_frame_pred_x = pred[frame_cnt][:nb_sed][sed_frame == 1]
        doa_frame_pred_y = pred[frame_cnt][nb_sed:2*nb_sed][sed_frame == 1]

        for cnt in range(nb_src_gt_list[frame_cnt]):
            doa_loss_gt += np.sqrt(
                (doa_frame_gt_x[cnt] - doa_frame_pred_x[cnt]) ** 2 +
                (doa_frame_gt_y[cnt] - doa_frame_pred_y[cnt]) ** 2
            )
            doa_loss_gt_cnt += 1

        # DOA Loss with respect to predicted confidence
        sed_frame_pred = pred_sed[frame_cnt]
        doa_frame_gt_x = gt[frame_cnt][:nb_sed][sed_frame_pred == 1]
        doa_frame_gt_y = gt[frame_cnt][nb_sed:2*nb_sed][sed_frame_pred == 1]

        doa_frame_pred_x = pred[frame_cnt][:nb_sed][sed_frame_pred == 1]
        doa_frame_pred_y = pred[frame_cnt][nb_sed:2*nb_sed][sed_frame_pred == 1]

        for cnt in range(nb_src_pred_list[frame_cnt]):
            doa_loss_pred += np.sqrt(
                (doa_frame_gt_x[cnt] - doa_frame_pred_x[cnt]) ** 2 +
                (doa_frame_gt_y[cnt] - doa_frame_pred_y[cnt]) ** 2
            )
            doa_loss_pred_cnt += 1

    if doa_loss_pred_cnt:
        doa_loss_pred /= doa_loss_pred_cnt

    if doa_loss_gt_cnt:
        doa_loss_gt /= doa_loss_gt_cnt

    max_nb_src_gt = np.max(nb_src_gt_list)
    conf_mat = confusion_matrix(nb_src_gt_list, nb_src_pred_list)
    conf_mat = conf_mat / (eps + np.sum(conf_mat, 1)[:, None].astype('float'))
    avg_accuracy = np.mean(np.diag(conf_mat[:max_nb_src_gt, :max_nb_src_gt]))  # In frames where more DOA's are
    # predicted, the conf_mat is no more square matrix, and the average skew's the results. Hence we always calculate
    # the accuracy wrt gt number of sources
    er_metric = [avg_accuracy, doa_loss_gt, doa_loss_pred, doa_loss_gt_cnt, doa_loss_pred_cnt, good_frame_cnt]
    return er_metric, conf_mat


def compute_doa_scores_regr_xyz(pred, gt, pred_sed, gt_sed):
    nb_src_gt_list = np.zeros(gt.shape[0]).astype(int)
    nb_src_pred_list = np.zeros(gt.shape[0]).astype(int)
    good_frame_cnt = 0
    less_frame_cnt = 0
    more_frame_cnt = 0
    doa_loss_gt = 0.0
    doa_loss_gt_cnt = 0
    doa_loss_pred = 0.0
    doa_loss_pred_cnt = 0
    nb_sed = gt_sed.shape[-1]

    for frame_cnt, sed_frame in enumerate(gt_sed):
        nb_src_gt_list[frame_cnt] = int(np.sum(sed_frame))
        nb_src_pred_list[frame_cnt] = int(np.sum(pred_sed[frame_cnt]))
        if nb_src_gt_list[frame_cnt] > nb_src_pred_list[frame_cnt]:
            less_frame_cnt = less_frame_cnt + 1
        elif nb_src_gt_list[frame_cnt] < nb_src_pred_list[frame_cnt]:
            more_frame_cnt = more_frame_cnt + 1
        else:
            good_frame_cnt = good_frame_cnt + 1

        # DOA Loss with respect to groundtruth
        doa_frame_gt_x = gt[frame_cnt][:nb_sed][sed_frame == 1]
        doa_frame_gt_y = gt[frame_cnt][nb_sed:2*nb_sed][sed_frame == 1]
        doa_frame_gt_z = gt[frame_cnt][2*nb_sed:][sed_frame == 1]

        doa_frame_pred_x = pred[frame_cnt][:nb_sed][sed_frame == 1]
        doa_frame_pred_y = pred[frame_cnt][nb_sed:2*nb_sed][sed_frame == 1]
        doa_frame_pred_z = pred[frame_cnt][2*nb_sed:][sed_frame == 1]

        for cnt in range(nb_src_gt_list[frame_cnt]):
            doa_loss_gt += np.sqrt(
                (doa_frame_gt_x[cnt] - doa_frame_pred_x[cnt]) ** 2 +
                (doa_frame_gt_y[cnt] - doa_frame_pred_y[cnt]) ** 2 +
                (doa_frame_gt_z[cnt] - doa_frame_pred_z[cnt]) ** 2
            )
            doa_loss_gt_cnt += 1

        # DOA Loss with respect to predicted confidence
        sed_frame_pred = pred_sed[frame_cnt]
        doa_frame_gt_x = gt[frame_cnt][:nb_sed][sed_frame_pred == 1]
        doa_frame_gt_y = gt[frame_cnt][nb_sed:2*nb_sed][sed_frame_pred == 1]
        doa_frame_gt_z = gt[frame_cnt][2*nb_sed:][sed_frame_pred == 1]

        doa_frame_pred_x = pred[frame_cnt][:nb_sed][sed_frame_pred == 1]
        doa_frame_pred_y = pred[frame_cnt][nb_sed:2*nb_sed][sed_frame_pred == 1]
        doa_frame_pred_z = pred[frame_cnt][2*nb_sed:][sed_frame_pred == 1]

        for cnt in range(nb_src_pred_list[frame_cnt]):
            doa_loss_pred += np.sqrt(
                (doa_frame_gt_x[cnt] - doa_frame_pred_x[cnt]) ** 2 +
                (doa_frame_gt_y[cnt] - doa_frame_pred_y[cnt]) ** 2 +
                (doa_frame_gt_z[cnt] - doa_frame_pred_z[cnt]) ** 2
            )
            doa_loss_pred_cnt += 1

    if doa_loss_pred_cnt:
        doa_loss_pred /= doa_loss_pred_cnt

    if doa_loss_gt_cnt:
        doa_loss_gt /= doa_loss_gt_cnt

    max_nb_src_gt = np.max(nb_src_gt_list)
    conf_mat = confusion_matrix(nb_src_gt_list, nb_src_pred_list)
    conf_mat = conf_mat / (eps + np.sum(conf_mat, 1)[:, None].astype('float'))
    avg_accuracy = np.mean(np.diag(conf_mat[:max_nb_src_gt, :max_nb_src_gt]))  # In frames where more DOA's are
    # predicted, the conf_mat is no more square matrix, and the average skew's the results. Hence we always calculate
    # the accuracy wrt gt number of sources
    er_metric = [avg_accuracy, doa_loss_gt, doa_loss_pred, doa_loss_gt_cnt, doa_loss_pred_cnt, good_frame_cnt]
    return er_metric, conf_mat
