# code in this file is adpated from
# https://github.com/iCGY96/ARPL
# https://github.com/wjun0830/Difficulty-Aware-Simulator

import torch
import numpy as np

from misc.util import *
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve

MAX_NUM = 999999

def compute_score(logit_list, softmax_list, score_wgts, branch_opt, fts=None):
    msp  = softmax_list[branch_opt].max(1)[0]
    mls  = logit_list[branch_opt].max(1)[0]
    if score_wgts[2] != 0:
        ftl = fts.mean(dim = [2,3]).norm(dim = 1, p = 2)
        temp = (score_wgts[0]*msp + score_wgts[1]*mls + score_wgts[2]*ftl)
    else:
        temp = (score_wgts[0]*msp + score_wgts[1]*mls)
    return temp


def evaluation(model, test_loader, out_loader, **options):
    model.eval()
    # torch.cuda.empty_cache()

    correct = 0
    total = 0
    n = 0

    pred_close = []
    pred_open = []
    labels_close = []
    labels_open = []
    score_close = []
    score_open = []

    open_labels = torch.zeros(MAX_NUM)
    probs = torch.zeros(MAX_NUM)
    
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data, labels
            batch_size = labels.size(0)
            output_dict = model(data, return_ft=True)
            logits_list = output_dict['logits']
            softmax_list = torch.stack(logits_list)
            softmax_list = torch.softmax(softmax_list / options['lgs_temp'], dim=2)
            if options['score_wgts'][2] != 0:
                fts = output_dict['fts']
                score_temp = compute_score(logits_list, softmax_list, options['score_wgts'], options['branch_opt'], fts=fts)
            else:
                score_temp = compute_score(logits_list, softmax_list, options['score_wgts'], options['branch_opt'])
            # import pdb;pdb.set_trace()
            score_close.append(score_temp.data.cpu().numpy())
            for b in range(batch_size):
                probs[n] = score_temp[b]
                open_labels[n] = labels[b].item()  # lưu đúng label (0–K-1)
                n += 1
            pred_label = softmax_list[options['branch_opt']].data.max(1)[1]
            total += labels.size(0)
            correct += (pred_label == labels.data).sum()
            # import pdb;pdb.set_trace()
            pred_close.append(softmax_list[options['branch_opt']].data.cpu().numpy())
            labels_close.append(labels.data.cpu().numpy())
        # import pdb;pdb.set_trace()
        for data, labels in out_loader:
            data, labels = data, labels
            batch_size = labels.size(0)
            output_dict = model(data, return_ft=True)
            logits_list = output_dict['logits']
            softmax_list = torch.stack(logits_list)
            softmax_list = torch.softmax(softmax_list / options['lgs_temp'], dim=2)
            if options['score_wgts'][2] != 0:
                fts = output_dict['fts']
                score_temp = compute_score(logits_list, softmax_list, options['score_wgts'], options['branch_opt'], fts=fts)
            else:
                score_temp = compute_score(logits_list, softmax_list, options['score_wgts'], options['branch_opt'])
            score_open.append(score_temp.data.cpu().numpy())
            for b in range(batch_size):
                probs[n] = score_temp[b]
                open_labels[n] = -1  # unknown label
                n += 1
            pred_open.append(softmax_list[options['branch_opt']].data.cpu().numpy())
            labels_open.append((torch.zeros_like(labels) - 1).cpu().numpy())
    import pdb;pdb.set_trace()

    acc = float(correct) * 100. / float(total)

    # Concatenate results
    pred_close = np.concatenate(pred_close, 0)
    pred_open = np.concatenate(pred_open, 0)
    labels_close = np.concatenate(labels_close, 0)
    labels_open = np.concatenate(labels_open, 0)
    score_close = np.concatenate(score_close, 0)
    score_open = np.concatenate(score_open, 0)

    # Final concatenations
    pred1 = np.argmax(pred_close, axis=1)
    pred2 = np.argmax(pred_open, axis=1)
    total_pred_label = np.concatenate([pred1, pred2], axis=0)
    total_label = np.concatenate([labels_close, labels_open], axis=0)
    total_score = np.concatenate([score_close, score_open], axis=0)

    # Threshold determination
    import pdb;pdb.set_trace()
    open_labels_bin = (open_labels[:n].cpu().numpy() != -1).astype(int)  # 1 = known, 0 = unknown
    prob = probs[:n].reshape(-1, 1)
    fpr, tpr, thresholds = roc_curve(open_labels_bin, prob)
    auroc = auc(fpr, tpr)
    thresh_idx = np.abs(tpr - 0.95).argmin()
    threshold = thresholds[thresh_idx]

    # Classification with rejection
    is_known_pred = (total_score >= threshold)
    pred_with_reject = np.where(is_known_pred, total_pred_label, -1)

    # Macro F1 across known + reject
    macro_f1 = f1_score(total_label, pred_with_reject, average='macro')

    # AUPR scores
    precision, recall, _ = precision_recall_curve(open_labels_bin, prob)
    aupr_in = auc(recall, precision)
    precision, recall, _ = precision_recall_curve(1 - open_labels_bin, -prob)
    aupr_out = auc(recall, precision)
    import pdb;pdb.set_trace()

    # ---------- Additional analysis ----------
    # Separate scores
    n_known = len(labels_close)
    n_unknown = len(labels_open)

    is_known = is_known_pred[:n_known]
    pred_known = pred_with_reject[:n_known]
    true_known = labels_close

    known_detected_total = np.sum(is_known)
    known_classify_correct = np.sum((is_known) & (pred_known == true_known))
    known_detect_rate = known_detected_total / n_known
    known_classify_acc = known_classify_correct / n_known

    is_unknown = is_known_pred[n_known:]
    unknown_reject_total = np.sum(~is_unknown)
    unknown_reject_rate = unknown_reject_total / n_unknown

    print('Accuracy (%): {:.3f}, AUROC: {:.5f}, AUPR_IN: {:.5f}, AUPR_OUT: {:.5f}, Macro F1-score: {:.5f}'.format(
        acc, auroc, aupr_in, aupr_out, macro_f1
    ))
    print('Known Detection Rate: {:.3f}, Known Classification Accuracy: {:.3f}, Unknown Rejection Rate: {:.3f}'.format(
        known_detect_rate, known_classify_acc, unknown_reject_rate
    ))

    return acc, auroc, aupr_in, aupr_out, macro_f1

