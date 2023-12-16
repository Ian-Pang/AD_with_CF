# pylint: disable=invalid-name
""" Reads in Calorimeter data from .hdf5 file and plots histograms of
    one particle type. Optionally, the geant reference and CaloGAN result
    can be plotted, too. Saves them as single files per plot.
    Use plot_all_particles.py to reproduce the plots of the papers.

    This code was used for the following publications:

    "CaloFlow: Fast and Accurate Generation of Calorimeter Showers with Normalizing Flows"
    by Claudius Krause and David Shih
    arxiv:2106.05285, Phys.Rev.D 107 (2023) 11, 113003

    "CaloFlow II: Even Faster and Still Accurate Generation of Calorimeter Showers with
     Normalizing Flows"
    by Claudius Krause and David Shih
    arXiv:2110.11377, Phys.Rev.D 107 (2023) 11, 113004

    Layout inspired by
    "CaloGAN: Simulating 3D High Energy Particle Showers in Multi-LayerElectromagnetic
     Calorimeters with Generative Adversarial Networks"
    by Michela Paganini, Luke de Oliveira, and Benjamin Nachman
    arxiv:1712.10321
    https://github.com/hep-lbdl/CaloGAN
"""

import argparse
import os

import torch
import numpy as np
import h5py

import plot_calo
from data import CaloDataset

parser = argparse.ArgumentParser()

parser.add_argument('--CaloFlow_file', '-f', help='path/to/CaloFlow_file.hdf5')
parser.add_argument('--results_dir', '-r', help='path/to/results_folder')
parser.add_argument('--folder_ref', help='path/to/GEANT_reference/')

parser.add_argument('--include_GEANT', action='store_true', help='Plot GEANT data, too')
parser.add_argument('--include_CaloGAN', action='store_true', help='Plot CaloGAN data, too')

parser.add_argument('--data_thres', type=float, default=0.,
                    help='Threshold in MeV to apply to CaloFlow data')
parser.add_argument('--GEANT_thres', type=float, default=0.,
                    help='Threshold in MeV to apply to GEANT data')

if __name__ == '__main__':
    args = parser.parse_args()

    if 'eplus' in args.CaloFlow_file:
        particle = 'eplus'
    elif 'gamma' in args.CaloFlow_file:
        particle = 'gamma'
    elif 'piplus' in args.CaloFlow_file:
        particle = 'piplus'
    else:
        question = 'Could not find "eplus", "gamma", or "piplus" in the CaloFlow_file. '
        question += 'Please specify the particle type. \n'
        particle = input(question)
        assert particle in ['eplus', 'gamma', 'piplus']

    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir)

    print("Plotting histograms for " + particle + " in folder " + args.results_dir)

    folder, _ = os.path.split(args.CaloFlow_file)
    file_name = os.path.split(os.path.splitext(args.CaloFlow_file)[0])[1]
    dataset = CaloDataset(folder, file_name, apply_logit=False)

    plot_ref = particle if args.include_GEANT else None
    plot_GAN = particle if args.include_CaloGAN else None

    num_events = len(dataset)
    layer_0 = dataset[:]['layer_0']*1e5
    layer_1 = dataset[:]['layer_1']*1e5
    layer_2 = dataset[:]['layer_2']*1e5
    all_layer = np.concatenate([layer_0.reshape(num_events, -1),
                                layer_1.reshape(num_events, -1),
                                layer_2.reshape(num_events, -1)], axis=1)
    energy = dataset[:]['energy'].reshape(num_events, 1)*1e2
    layer_0 = torch.from_numpy(layer_0)
    layer_1 = torch.from_numpy(layer_1)
    layer_2 = torch.from_numpy(layer_2)
    all_layer = torch.from_numpy(all_layer)
    energy = torch.from_numpy(energy)

    if particle == 'eplus':
        vmin = [3e-2, 2.5e-1, 2.5e-2]
        vmax = [3e3, 9.2e3, 1.2e1]
    elif particle == 'gamma':
        vmin = [3e-2, 2.5e-1, 3e-2]
        vmax = [2.2e3, 9.8e3, 1.7e1]
    else:
        vmin = [3.5e-2, 6.5e-1, 2e-1]
        vmax = [1.6e2, 2.1e3, 2.4e2]

    for plot_layer_id, sample in enumerate([layer_0, layer_1, layer_2]):
        filename = 'avg_layer_'+str(plot_layer_id) +'.pdf'
        plot_calo.plot_average_voxel(sample,
                                     os.path.join(args.results_dir, filename),
                                     plot_layer_id,
                                     vmin=vmin[plot_layer_id],
                                     vmax=vmax[plot_layer_id],
                                     lower_threshold=args.data_thres)
        filename = 'generated_samples_E_layer_'+str(plot_layer_id) + '.png'
        plot_calo.plot_layer_energy(sample,
                                    os.path.join(args.results_dir, filename),
                                    plot_layer_id, plot_ref=plot_ref,
                                    plot_GAN=plot_GAN, lower_threshold=args.data_thres,
                                    ref_thres=args.GEANT_thres)

        filename = 'sparsity_layer_'+str(plot_layer_id) + '.png'
        plot_calo.plot_layer_sparsity(sample,
                                      os.path.join(args.results_dir, filename),
                                      plot_layer_id, plot_ref=plot_ref,
                                      plot_GAN=plot_GAN, ref_thres=args.GEANT_thres,
                                      threshold=args.data_thres)
        filename = 'e_ratio_layer_'+str(plot_layer_id) + '.png'
        plot_calo.plot_layer_E_ratio(sample,
                                     os.path.join(args.results_dir, filename),
                                     plot_layer_id, plot_ref=plot_ref, ref_thres=args.GEANT_thres,
                                     plot_GAN=plot_GAN, lower_threshold=args.data_thres)
        for which_voxel in [1, 2, 3, 4, 5]:
            filename = str(which_voxel)+'_brightest_voxel_layer_'+str(plot_layer_id) + '.png'
            plot_calo.plot_brightest_voxel(sample,
                                           os.path.join(args.results_dir, filename),
                                           plot_layer_id, which_voxel=which_voxel,
                                           plot_ref=plot_ref, plot_GAN=plot_GAN,
                                           lower_threshold=args.data_thres,
                                           ref_thres=args.GEANT_thres)
        filename = 'lateral_width_layer_'+str(plot_layer_id) + '.png'
        plot_calo.plot_layer_lateral_width(sample,
                                           os.path.join(args.results_dir, filename),
                                           plot_layer_id,
                                           plot_ref=plot_ref, plot_GAN=plot_GAN,
                                           lower_threshold=args.data_thres,
                                           ref_thres=args.GEANT_thres)
    filename = 'generated_samples_E_total' + '.png'
    plot_calo.plot_total_energy(all_layer,
                                os.path.join(args.results_dir, filename),
                                plot_ref=plot_ref, plot_GAN=plot_GAN,
                                ref_thres=args.GEANT_thres,
                                lower_threshold=args.data_thres)

    for use_log in [False]: #True
        filename = 'energy_distribution_layer' + (use_log)*'_log' +'.png'
        plot_calo.plot_energy_distribution_layer(all_layer, energy,
                                                 os.path.join(args.results_dir, filename),
                                                 lower_threshold=args.data_thres,
                                                 use_log=use_log)
        filename = 'energy_distribution_total' + (use_log)*'_log' '.png'
        plot_calo.plot_energy_distribution_total(all_layer, energy,
                                                 os.path.join(args.results_dir, filename),
                                                 lower_threshold=args.data_thres,
                                                 use_log=use_log)

    for layer in [0, 1, 2]:
        for use_log in [False, True]:
            filename = 'energy_fraction_layer_'+str(layer)+ (use_log)*'_log'+ '.png'
            plot_calo.plot_energy_fraction(all_layer,
                                           os.path.join(args.results_dir, filename),
                                           layer, plot_ref=plot_ref,
                                           plot_GAN=plot_GAN,
                                           use_log=use_log,
                                           lower_threshold=args.data_thres,
                                           ref_thres=args.GEANT_thres)
    filename = 'shower_depth.png'
    plot_calo.plot_shower_depth(all_layer,
                                os.path.join(args.results_dir, filename),
                                plot_ref=plot_ref, plot_GAN=plot_GAN,
                                lower_threshold=args.data_thres,
                                ref_thres=args.GEANT_thres)
    filename = 'depth_weighted_total_energy.png'
    plot_calo.plot_depth_weighted_total_energy(all_layer,
                                               os.path.join(args.results_dir, filename),
                                               plot_ref=plot_ref, plot_GAN=plot_GAN,
                                               lower_threshold=args.data_thres,
                                               ref_thres=args.GEANT_thres)
    filename = 'depth_weighted_energy_normed.png'
    plot_calo.plot_depth_weighted_energy_normed(all_layer,
                                                os.path.join(args.results_dir, filename),
                                                plot_ref=plot_ref, plot_GAN=plot_GAN,
                                                lower_threshold=args.data_thres,
                                                ref_thres=args.GEANT_thres)
    filename = 'depth_weighted_energy_normed_std.png'
    plot_calo.plot_depth_weighted_energy_normed_std(all_layer,
                                                    os.path.join(args.results_dir, filename),
                                                    plot_ref=plot_ref, plot_GAN=plot_GAN,
                                                    lower_threshold=args.data_thres,
                                                    ref_thres=args.GEANT_thres)
    for scan_dir in ['eta', 'phi']:
        filename = 'centroid_corr_0_1_'+scan_dir+'.png'
        plot_calo.plot_centroid_correlation(all_layer,
                                            os.path.join(args.results_dir, filename),
                                            0, 1, scan=scan_dir,
                                            plot_ref=plot_ref, plot_GAN=plot_GAN,
                                            lower_threshold=args.data_thres,
                                            ref_thres=args.GEANT_thres)
        filename = 'centroid_corr_1_2_'+scan_dir+'.png'
        plot_calo.plot_centroid_correlation(all_layer,
                                            os.path.join(args.results_dir, filename),
                                            1, 2, scan=scan_dir,
                                            plot_ref=plot_ref, plot_GAN=plot_GAN,
                                            lower_threshold=args.data_thres,
                                            ref_thres=args.GEANT_thres)
        filename = 'centroid_corr_0_2_'+scan_dir+'.png'
        plot_calo.plot_centroid_correlation(all_layer,
                                            os.path.join(args.results_dir, filename),
                                            0, 2, scan=scan_dir,
                                            plot_ref=plot_ref, plot_GAN=plot_GAN,
                                            lower_threshold=args.data_thres,
                                            ref_thres=args.GEANT_thres)

    #filename = 'nearest_neighbor_all_layers.pdf'
    #plot_calo.plot_nn(all_layer, energy,
    #                  os.path.join(args.results_dir, filename),
    #                  layer='all',
    #                  num_events='fixed',
    #                  ref_data_path=args.folder_ref,
    #                  ref_data_name='train_CaloGAN_'+particle,
    #                  lower_threshold=args.data_thres)
