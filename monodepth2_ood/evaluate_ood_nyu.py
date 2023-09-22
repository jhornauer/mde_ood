from __future__ import absolute_import, division, print_function

import os
import copy
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import networks
import id_datasets as datasets
from extended_options import OODOptions
import monodepth2.networks as legacy
import progressbar
import matplotlib.pyplot as plt
import ood_datasets as ood_datasets

from sklearn.metrics import roc_auc_score, average_precision_score

from networks import BayesCap

np.random.seed(0)
torch.manual_seed(0)


def get_metrics_ood(label, score, invert_score=False):
    results_dict = {}
    if invert_score:
        score = score * (-1)

    error = 1 - label
    rocauc = roc_auc_score(label, score)

    aupr_success = average_precision_score(label, score)
    aupr_errors = average_precision_score(error, (1 - score))

    # calculate fpr @ 95% tpr
    fpr = 0
    eval_range = np.arange(score.min(), score.max(), (score.max() - score.min()) / 10000)
    for i, delta in enumerate(eval_range):
        tpr = len(score[(label == 1) & (score >= delta)]) / len(score[(label == 1)])
        fpr = len(score[(error == 1) & (score >= delta)]) / len(score[(error == 1)])
        if 0.9505 >= tpr >= 0.9495:
            fpr = len(score[(error == 1) & (score >= delta)]) / len(score[(error == 1)])
            break
        if tpr < 0.9495:
            fpr = len(score[(error == 1) & (score >= delta)]) / len(score[(error == 1)])
            break
    if fpr == 0:
        fpr = len(score[(error == 1) & (score >= eval_range[-1])]) / len(score[(error == 1)])

    results_dict["rocauc"] = rocauc
    results_dict["aupr_success"] = aupr_success
    results_dict["aupr_error"] = aupr_errors
    results_dict["fpr"] = fpr
    return results_dict


def post_process_uct(opt, enc, dec, loader, prefix, plotting):
    pred_ucts = []
    bar = progressbar.ProgressBar(max_value=len(loader))
    for i, data in enumerate(loader):
        rgb_img = data[("color", 0, 0)].cuda()

        # updating progress bar
        bar.update(i)

        # post-processed results require each image to have two forward passes
        rgb_img = torch.cat((rgb_img, torch.flip(rgb_img, [3])), 0)

        with torch.no_grad():
            output = dec(enc(rgb_img))
            pred_depth = output[("depth", 0)][:, 0]
            pred_depth = pred_depth.cpu().numpy()

        # applying Monodepthv1 post-processing to improve depth and get uncertainty
        N = pred_depth.shape[0] // 2
        pred_uncert = np.abs(pred_depth[:N] - pred_depth[N:, :, ::-1])
        if plotting:
            plt.imsave(os.path.join(opt.output_dir, prefix + '_{:06d}_post_uct.png'.format(i)), pred_uncert[0],
                       cmap='hot')
        pred_ucts.append(pred_uncert)
    return np.concatenate(pred_ucts)


def drop_uct(opt, enc, dec, loader, prefix, plotting):
    pred_ucts = []
    bar = progressbar.ProgressBar(max_value=len(loader))
    for i, data in enumerate(loader):
        rgb_img = data[("color", 0, 0)].cuda()

        # updating progress bar
        bar.update(i)

        with torch.no_grad():
            # infer multiple predictions from multiple networks with dropout
            depth_distribution = []
            # we infer 8 predictions
            for j in range(8):
                output = dec(enc(rgb_img))
                depth_distribution.append(torch.unsqueeze(output[("depth", 0)], 0))
            depth_distribution = torch.cat(depth_distribution, 0)

        # uncertainty as variance of the predictions
        pred_uncert = torch.var(depth_distribution, dim=0, keepdim=False).cpu()[:, 0].numpy()
        pred_uncert = (pred_uncert - np.min(pred_uncert)) / (np.max(pred_uncert) - np.min(pred_uncert))
        if plotting:
            plt.imsave(os.path.join(opt.output_dir, prefix + '_{:06d}_drop_uct.png'.format(i)), pred_uncert[0],
                       cmap='hot')
        pred_ucts.append(pred_uncert)
    return np.concatenate(pred_ucts)


def log_uct(opt, enc, dec, loader, prefix, plotting):
    pred_ucts = []
    bar = progressbar.ProgressBar(max_value=len(loader))
    for i, data in enumerate(loader):
        rgb_img = data[("color", 0, 0)].cuda()

        # updating progress bar
        bar.update(i)

        with torch.no_grad():
            features = enc(rgb_img)
            output = dec(features)
            # only needed is maps are saved

            pred_uncert = torch.exp(output[("uncert", 0)])[:, 0].cpu().numpy()
            pred_uncert = (pred_uncert - np.min(pred_uncert)) / (np.max(pred_uncert) - np.min(pred_uncert))
            if plotting:
                plt.imsave(os.path.join(opt.output_dir, prefix + '_{:06d}_log_uct.png'.format(i)), pred_uncert[0],
                           cmap='hot')
            pred_ucts.append(pred_uncert)
    return np.concatenate(pred_ucts)


def autoencoder_uct(opt, enc, rgb_dec, loader, prefix, plotting):
    crit = nn.L1Loss(reduction='none')
    pred_ucts = []
    bar = progressbar.ProgressBar(max_value=len(loader))
    for i, data in enumerate(loader):
        imgs = data[("color", 0, 0)].cuda()

        bar.update(i)
        with torch.no_grad():
            # get depth prediction
            features = enc(imgs)
            output_rgb = rgb_dec(features)
            pred_img = output_rgb[("rgb", 0)]
            pred_uncert = np.abs(crit(pred_img, imgs).cpu().numpy())
            pred_uncert = np.max(pred_uncert, axis=1)
            pred_uncert = (pred_uncert - np.min(pred_uncert)) / (np.max(pred_uncert) - np.min(pred_uncert))
            if plotting:
                plt.imsave(os.path.join(opt.output_dir, prefix + '_{:06d}_img.png'.format(i)),
                           np.transpose(imgs[0].cpu().numpy(), (1, 2, 0)), vmin=0.0, vmax=1.0)
                plt.imsave(os.path.join(opt.output_dir, prefix + '_{:06d}_pred.png'.format(i)),
                           np.transpose(pred_img[0].clip(0, 1).cpu().numpy(), (1, 2, 0)), vmin=0.0, vmax=1.0)
                plt.imsave(os.path.join(opt.output_dir, prefix + '_{:06d}_imgdecoder_uct.png'.format(i)),
                           pred_uncert[0], cmap='hot')
            pred_ucts.append(pred_uncert)
    return np.concatenate(pred_ucts)


def bayescap_uct(opt, enc, dec, bcap, loader, prefix, plotting):
    pred_ucts = []
    bar = progressbar.ProgressBar(max_value=len(loader))
    for i, data in enumerate(loader):
        imgs = data[("color", 0, 0)].cuda()

        bar.update(i)
        with torch.no_grad():
            # get depth prediction
            output_depth = dec(enc(imgs))
            pred_depth = output_depth[("depth", 0)]
            x = pred_depth / opt.max_depth
            x_mu, x_alpha, x_beta = bcap(x)

            a_map = (1 / (x_alpha[0] + 1e-5)).to('cpu').data
            b_map = x_beta[0].to('cpu').data
            x_var = (a_map**2)*(torch.exp(torch.lgamma(3/(b_map + 1e-2)))/torch.exp(torch.lgamma(1/(b_map + 1e-2))))
            pred_uncert = x_var.cpu().numpy()
            pred_uncert[np.isnan(pred_uncert)] = np.nanmin(pred_uncert)
            pred_uncert[np.isinf(pred_uncert)] = np.nanmax(pred_uncert[np.isfinite(pred_uncert)])
            pred_ucts.append(copy.deepcopy(pred_uncert))
            if plotting:
                percentile99 = np.percentile(pred_uncert, 99)
                percentile01 = np.percentile(pred_uncert, 1)
                pred_uncert[pred_uncert <= percentile01] = percentile01
                pred_uncert[pred_uncert >= percentile99] = percentile99
                plt.imsave(os.path.join(opt.output_dir, prefix + '_bayescap_{:06d}_uct.png'.format(i)), pred_uncert[0],
                           cmap='hot')
    return np.concatenate(pred_ucts)


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    opt.max_depth = 10.0
    opt.batch_size = 1

    print("-> Beginning inference...")

    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
    assert os.path.isdir(opt.load_weights_folder), "Cannot find a folder at {}".format(opt.load_weights_folder)

    if not os.path.exists(opt.output_dir) and opt.plot_results:
        os.makedirs(opt.output_dir)

    print("-> Loading weights from {}".format(opt.load_weights_folder))

    # prepare just a single path
    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
    encoder_dict = torch.load(encoder_path)
    height = encoder_dict['height']
    width = encoder_dict['width']

    id_dataset = datasets.NYUDataset(opt.data_path + '/val/', split='val', height=height, width=width)
    id_dataloader = DataLoader(id_dataset, opt.batch_size, shuffle=False, pin_memory=True, drop_last=False)

    if opt.ood_dataset == "places365":
        ood_dataset = ood_datasets.Places365Dataset(os.path.join(opt.ood_data, 'train_256_places365standard'),
                                                    height=height, width=width)
    else:
        raise NotImplementedError
    ood_dataloader = DataLoader(ood_dataset, opt.batch_size, shuffle=False, pin_memory=True, drop_last=False)

    # load a single encoder and decoder
    encoder = legacy.ResnetEncoder(opt.num_layers, False)
    if not opt.autoencoder:
        depth_decoder = networks.DepthUncertaintyDecoder_Supervised(encoder.num_ch_enc, scales=opt.scales,
                                                                    dropout=opt.dropout, uncert=opt.uncert)
        depth_decoder.load_state_dict(torch.load(decoder_path))
        depth_decoder.cuda()
        depth_decoder.eval()
    else:
        depth_decoder = None
    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    encoder.cuda()
    encoder.eval()

    print("-> Computing predictions with size {}x{}".format(width, height))

    if opt.post_process:
        ucts_id = post_process_uct(opt, encoder, depth_decoder, id_dataloader, prefix='id', plotting=opt.plot_results)
        ucts_ood = post_process_uct(opt, encoder, depth_decoder, ood_dataloader, prefix='ood', plotting=opt.plot_results)
    elif opt.dropout:
        ucts_id = drop_uct(opt, encoder, depth_decoder, id_dataloader, prefix='id', plotting=opt.plot_results)
        ucts_ood = drop_uct(opt, encoder, depth_decoder, ood_dataloader, prefix='ood', plotting=opt.plot_results)
    elif opt.log:
        ucts_id = log_uct(opt, encoder, depth_decoder, id_dataloader, prefix='id', plotting=opt.plot_results)
        ucts_ood = log_uct(opt, encoder, depth_decoder, ood_dataloader, prefix='ood', plotting=opt.plot_results)
    elif opt.autoencoder:
        rgb_decoder = networks.ImageDecoder_Supervised(encoder.num_ch_enc)
        rgb_decoder_path = os.path.join(opt.load_weights_folder, "image.pth")
        rgb_decoder.load_state_dict(torch.load(rgb_decoder_path))
        rgb_decoder.cuda()
        rgb_decoder.eval()
        ucts_id = autoencoder_uct(opt, encoder, rgb_decoder, id_dataloader, prefix='id', plotting=opt.plot_results)
        ucts_ood = autoencoder_uct(opt, encoder, rgb_decoder, ood_dataloader, prefix='ood', plotting=opt.plot_results)
    elif opt.bayescap:
        bayescap = BayesCap()
        bayescap_path = os.path.join(opt.load_weights_folder, "bayescap.pth")
        bayescap.load_state_dict(torch.load(bayescap_path))
        bayescap.cuda()
        bayescap.eval()
        ucts_id = bayescap_uct(opt, encoder, depth_decoder, bayescap, id_dataloader, prefix='id',
                               plotting=opt.plot_results)
        ucts_ood = bayescap_uct(opt, encoder, depth_decoder, bayescap, ood_dataloader, prefix='ood',
                                plotting=opt.plot_results)
    else:
        raise Exception('option not available')

    ood_scores_id = np.mean(ucts_id, axis=(1, 2))
    ood_scores_ood = np.mean(ucts_ood, axis=(1, 2))

    labels = np.concatenate([np.ones(len(id_dataset)), np.zeros(len(ood_dataset))], axis=0)
    scores = np.concatenate([ood_scores_id, ood_scores_ood], axis=0)

    results_dict = get_metrics_ood(labels, scores, invert_score=True)
    print('\n')
    print('AUROC: \t \t \t {:.2%}'.format(results_dict["rocauc"]))
    print('AUPR-Success: \t \t {:.2%}'.format(results_dict["aupr_success"]))
    print('AUPR-Error: \t \t {:.2%}'.format(results_dict["aupr_error"]))
    print('FPRR at 95% TPR: \t {:.2%}'.format(results_dict["fpr"]))

    # see you next time!
    print("\n-> Done!")


if __name__ == "__main__":
    warnings.simplefilter("ignore", UserWarning)
    options = OODOptions()
    evaluate(options.parse())
