# pylint: disable=invalid-name
""" Reads in Calorimeter data files (.hdf5) for up to 3 particle types and plots them
    together in histograms. Also supports plotting of samples from the student model
    for comparison. The produced files are the ones used for the publications.
    Layout closely follows the one of plot_calo.py and plot_my_calo.py, which
    plot only one particle at a time.

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

import numpy as np
import pandas as pd
import h5py
import matplotlib
import matplotlib.pyplot as plt

import plotting_helper as plthlp

parser = argparse.ArgumentParser()

parser.add_argument('--eplus_file', default=None, help='path/to/CaloFlow_eplus.hdf5')
parser.add_argument('--gamma_file', default=None, help='path/to/CaloFlow_gamma.hdf5')
parser.add_argument('--piplus_file', default=None, help='path/to/CaloFlow_piplus.hdf5')
parser.add_argument('--eplus_student_file', default=None,
                    help='path/to/CaloFlow_student_eplus.hdf5')
parser.add_argument('--gamma_student_file', default=None,
                    help='path/to/CaloFlow_student_gamma.hdf5')
parser.add_argument('--piplus_student_file', default=None,
                    help='path/to/CaloFlow_student_piplus.hdf5')
parser.add_argument('--results_dir', '-r', help='path/to/results_folder')


parser.add_argument('--data_thres', type=float, default=0.,
                    help='Threshold in MeV to apply to CaloFlow data')
parser.add_argument('--GEANT_thres', type=float, default=0.,
                    help='Threshold in MeV to apply to GEANT data')
parser.add_argument('--num_events', '-n', default=None, help='Use smaller dataset to save memory')

parser.add_argument('--show', action='store_true', help='Only show and not save plots')
parser.add_argument('--plot_GAN', action='store_true', help='plot CaloGAN, too')
parser.add_argument('--no_legend', action='store_true', help='do not include legends')

INPUT_DIMS = {'0': (3, 96), '1': (12, 12), '2': (12, 6)}
INPUT_SIZE = {'0': 288, '1': 144, '2': 72}

# CaloGAN color scheme:
colors = matplotlib.cm.gnuplot2(np.linspace(0.2, 0.8, 3))
COLORS = {'eplus': colors[0], 'gamma': colors[1], 'piplus': colors[2]}
del colors

# legend labels:
GAN_legend_dict = {'eplus': r'$e^+$ CaloGAN', 'gamma': r'$\gamma$ CaloGAN',
                   'piplus': r'$\pi^+$ CaloGAN'}
GEANT_legend_dict = {'eplus': r'$e^+$ GEANT', 'gamma': r'$\gamma$ GEANT',
                     'piplus': r'$\pi^+$ GEANT'}
Flow_legend_dict = {'eplus': r'$e^+$ CaloFlow', 'gamma': r'$\gamma$ CaloFlow',
                    'piplus': r'$\pi^+$ CaloFlow'}
# global switch if used for v1 or v2 (set to v2 as soon as student file is given, v1 else)
IS_v2 = False

def plot_total_energy(particle_list, which_list, layer_0_list, layer_1_list, layer_2_list, filename,
                      save_it=True, plot_GAN=False, flow_thres=0., geant_thres=0., is_sub=False):
    """ plots the total energy deposit in all 3 layers for a given batch
        particle_list (list of str): list of particles to be plotted, must contain
                                     "eplus", "gamma", "piplus"
        which_list (list of str): list of "teacher" or "student" to identify flow
        layer_0_list (list of np.array): list of len(particle_list) containing layer 0 data
        layer_1_list (list of np.array): list of len(particle_list) containing layer 1 data
        layer_2_list (list of np.array): list of len(particle_list) containing layer 2 data
        filename (string): filepath including filename with extension to save image
        save_it (bool): if True, plot will be saved to fp, if not, plot is only shown
        plot_GAN (bool): if True, results from our CaloGAN will be plotted, too
        flow_thres (float): threshold to be applied to flow data before plotting
        geant_thres (float): threshold to be applied to GEANT4 reference data
        is_sub: bool, whether or not plot is stand-alone or subfigure of a larger figure
    """
    assert len(particle_list) == len(layer_0_list)
    assert len(particle_list) == len(layer_1_list)
    assert len(particle_list) == len(layer_2_list)

    bins = np.linspace(0, 120, 50)
    if not is_sub:
        plt.figure(figsize=(10, 10))

    for i, particle in enumerate(particle_list):
        layer_0_list[i] = plthlp.thres_clip(layer_0_list[i], flow_thres)
        layer_1_list[i] = plthlp.thres_clip(layer_1_list[i], flow_thres)
        layer_2_list[i] = plthlp.thres_clip(layer_2_list[i], flow_thres)
        assert (layer_0_list[i].shape[-1] == 288) and\
            (layer_1_list[i].shape[-1] == 144) and\
            (layer_2_list[i].shape[-1] == 72), (
                "Are you sure the input is of the right form?")
        energies = plthlp.energy_sum(layer_0_list[i]) +\
            plthlp.energy_sum(layer_1_list[i]) + plthlp.energy_sum(layer_2_list[i])

        if which_list[i] == 'teacher':
            if geant_thres not in [0., 1e-1, 1e-2, 1e-3, 1e-4]:
                raise ValueError(
                    "geant_thres {} not in pre-computed set".format(geant_thres))
            reference_data = pd.read_hdf('plots_reference_{:1.0e}.hdf'.format(geant_thres))
            reference = reference_data.loc[particle]['energy_layer_0_{:1.0e}'.format(geant_thres)]
            reference = reference + reference_data.loc[particle]\
                ['energy_layer_1_{:1.0e}'.format(geant_thres)]
            reference = reference + reference_data.loc[particle]\
                ['energy_layer_2_{:1.0e}'.format(geant_thres)]
            color = COLORS[particle]
            label = GEANT_legend_dict[particle]
            _ = plt.hist(reference / 1000., bins=bins, histtype='stepfilled',
                         linewidth=2, alpha=0.2, density=True, color=color,
                         label=label)
            if plot_GAN:
                reference_data = pd.read_hdf('plots_reference_CaloGAN.hdf')
                reference = reference_data.loc[particle]['energy_layer_0']
                reference = reference + reference_data.loc[particle]['energy_layer_1']
                reference = reference + reference_data.loc[particle]['energy_layer_2']
                label = GAN_legend_dict[particle]
                _ = plt.hist(reference / 1000., bins=bins, histtype='step',
                             linewidth=2, alpha=0.6, density=True, color=color,
                             label=label, linestyle='dashed')

        flow_label = Flow_legend_dict[particle]
        if IS_v2:
            style_dict = {'label': flow_label+ ' ' + which_list[i],
                          'linestyle': 'dashdot' if which_list[i] == 'teacher' else 'solid',
                          'alpha': 1 if which_list[i] == 'student' else 0.5}
        else:
            style_dict = {'label': flow_label, 'linestyle': 'solid', 'alpha': 1}
        _ = plt.hist(energies, bins=bins, histtype='step', linewidth=3,
                     color=color, density='True', **style_dict)

    plt.yscale('log')
    plt.ylim([5e-6, 2e-1])
    if not args.no_legend:
        if IS_v2:
            entries = 2 if plot_GAN else 3
            new_leg_handles, new_leg_labels = plt.gca().get_legend_handles_labels()
        else:
            entries = 3 if plot_GAN else 2
            new_leg_handles, new_leg_labels = reorder_legend(entries, len(particles))
        plt.legend(new_leg_handles, new_leg_labels, fontsize=(16 if plot_GAN else 20), ncol=entries)

    plt.xlabel(r'$\hat{E}_\mathrm{tot}$ (GeV)')

    if not is_sub:

        plt.tight_layout()
        if save_it:
            plt.savefig(filename, dpi=300)
            plt.close()
        else:
            plt.show()

def plot_layer_energy(particle_list, which_list, layer_list, filename, layer,
                      save_it=True, plot_GAN=False, flow_thres=0., geant_thres=0., is_sub=False):
    """ plots the energy deposit in the full layer for a given batch
        particle_list (list of str): list of particles to be plotted, must contain
                                     "eplus", "gamma", "piplus"
        which_list (list of str): list of "teacher" or "student" to identify flow
        layer_list (list of np.array): list of len(particle_list) containing the data
        filename (string): filepath including filename with extension to save image
        layer (int): which layer (0, 1, 2) the data is from
        save_it (bool): if True, plot will be saved to fp, if not, plot is only shown
        plot_GAN (bool): if True, results from our CaloGAN will be plotted, too
        flow_thres (float): threshold to be applied to flow data before plotting
        geant_thres (float): threshold to be applied to GEANT4 reference data
        is_sub: bool, whether or not plot is stand-alone or subfigure of a larger figure
    """
    assert len(particle_list) == len(layer_list)

    axis_label = r'''$E_{%(layer_id)d}$ (GeV)'''
    axis_label = axis_label % {'layer_id': layer}

    if layer == 0:
        bins = np.logspace(-2, 2, 100)
        x_max = 40
    elif layer == 1:
        bins = np.logspace(-1, 3, 100)
        x_max = 140
    elif layer == 2:
        bins = np.logspace(-2, 2, 100)
        x_max = 100
    if not is_sub:
        plt.figure(figsize=(10, 10))

    for i, particle in enumerate(particle_list):
        color = COLORS[particle]

        layer_list[i] = plthlp.thres_clip(layer_list[i], flow_thres)
        energies = plthlp.energy_sum(layer_list[i])

        if which_list[i] == 'teacher':
            if geant_thres not in [0., 1e-1, 1e-2, 1e-3, 1e-4]:
                raise ValueError(
                    "geant_thres {} not in pre-computed set.".format(geant_thres))
            reference_data = pd.read_hdf('plots_reference_{:1.0e}.hdf'.format(geant_thres))
            reference = reference_data.loc[particle]\
                ['energy_layer_{}_{:1.0e}'.format(str(layer), geant_thres)]
            label = GEANT_legend_dict[particle]
            _ = plt.hist(reference / 1000., bins=bins, histtype='stepfilled',
                         linewidth=2, alpha=0.2, density=True, color=color,
                         label=label)
            if plot_GAN:
                reference_data = pd.read_hdf('plots_reference_CaloGAN.hdf')
                reference = reference_data.loc[particle]['energy_layer_{}'.format(str(layer))]
                label = GAN_legend_dict[particle]
                _ = plt.hist(reference / 1000., bins=bins, histtype='step',
                             linewidth=2., alpha=0.6, density=True, color=color,
                             label=label, linestyle='dashed')

        flow_label = Flow_legend_dict[particle]
        if IS_v2:
            style_dict = {'label': flow_label+ ' ' + which_list[i],
                          'linestyle': 'dashdot' if which_list[i] == 'teacher' else 'solid',
                          'alpha': 1 if which_list[i] == 'student' else 0.5}
        else:
            style_dict = {'label': flow_label, 'linestyle': 'solid', 'alpha': 1}

        _ = plt.hist(energies, bins=bins, histtype='step', linewidth=3,
                     color=color, density='True', **style_dict)


    plt.yscale('log')
    plt.xscale('log')
    plt.xlim(xmax=x_max)
    plt.ylim([1e-6, 5e1])
    if not args.no_legend:
        if IS_v2:
            entries = 2 if plot_GAN else 3
            new_leg_handles, new_leg_labels = plt.gca().get_legend_handles_labels()
        else:
            entries = 3 if plot_GAN else 2
            new_leg_handles, new_leg_labels = reorder_legend(entries, len(particles))
        plt.legend(new_leg_handles, new_leg_labels, fontsize=(16 if plot_GAN else 20), ncol=entries)

    plt.xlabel(axis_label)

    if not is_sub:
        plt.tight_layout()
        if save_it:
            plt.savefig(filename, dpi=300)
            plt.close()
        else:
            plt.show()

def plot_energy_fraction(particle_list, which_list, layer_0_list, layer_1_list, layer_2_list,
                         filename, layer, save_it=True, plot_GAN=False, use_log=False,
                         flow_thres=0., geant_thres=0., is_sub=False):
    """ plots the energy fraction deposited in the given layer for a given batch
        which_list (list of str): list of "teacher" or "student" to identify flow
        layer: str (0, 1, or 2) to indicate layer number
        save_it: bool, whether or not to save (or just show) result
    """
    assert len(particle_list) == len(layer_0_list)
    assert len(particle_list) == len(layer_1_list)
    assert len(particle_list) == len(layer_2_list)

    if not is_sub:
        plt.figure(figsize=(10, 10))

    axis_label = r'''$E_{%(layer_id)d} / \hat{E}_\mathrm{tot}$'''
    axis_label = axis_label % {'layer_id': layer}

    for i, particle in enumerate(particle_list):
        assert (layer_0_list[i].shape[-1] == 288) and\
            (layer_1_list[i].shape[-1] == 144) and\
            (layer_2_list[i].shape[-1] == 72), (
                "Are you sure the input is of the right form?")
        color = COLORS[particle]
        layer_0_list[i] = plthlp.thres_clip(layer_0_list[i], flow_thres)
        layer_1_list[i] = plthlp.thres_clip(layer_1_list[i], flow_thres)
        layer_2_list[i] = plthlp.thres_clip(layer_2_list[i], flow_thres)
        energies_0 = plthlp.energy_sum(layer_0_list[i])
        energies_1 = plthlp.energy_sum(layer_1_list[i])
        energies_2 = plthlp.energy_sum(layer_2_list[i])
        energies = energies_0 + energies_1 + energies_2

        if layer == 0:
            plot_energy = energies_0
            if use_log:
                bins = np.logspace(-4, 0, 100)
            else:
                bins = np.linspace(0., 0.4, 100)
        elif layer == 1:
            plot_energy = energies_1
            if use_log:
                bins = np.logspace(-1, 0, 100)
            else:
                bins = np.linspace(0., 1., 100)
        elif layer == 2:
            plot_energy = energies_2
            if use_log:
                bins = np.logspace(-4, 1, 100)
            else:
                bins = np.linspace(0., 0.008, 100)

        if which_list[i] == 'teacher':
            if geant_thres not in [0., 1e-1, 1e-2, 1e-3, 1e-4]:
                raise ValueError(
                    "geant_thres {} not in pre-computed set.".format(geant_thres))
            reference_data = pd.read_hdf('plots_reference_{:1.0e}.hdf'.format(geant_thres))
            reference = reference_data.loc[particle]\
                ['energy_layer_{}_{:1.0e}'.format(str(layer), geant_thres)]

            reference_total = reference_data.loc[particle]\
                ['energy_layer_0_{:1.0e}'.format(geant_thres)]
            reference_total = reference_total + reference_data.loc[particle]\
                ['energy_layer_1_{:1.0e}'.format(geant_thres)]
            reference_total = reference_total + reference_data.loc[particle]\
                ['energy_layer_2_{:1.0e}'.format(geant_thres)]

            label = GEANT_legend_dict[particle]

            _ = plt.hist(reference / reference_total, bins=bins, histtype='stepfilled',
                         linewidth=2, alpha=0.2, density=True, color=color,
                         label=label)
            if plot_GAN:
                reference_data = pd.read_hdf('plots_reference_CaloGAN.hdf')
                reference = reference_data.loc[particle]['energy_layer_{}'.format(str(layer))]

                reference_total = reference_data.loc[particle]['energy_layer_0']
                reference_total = reference_total + reference_data.loc[particle]['energy_layer_1']
                reference_total = reference_total + reference_data.loc[particle]['energy_layer_2']
                label = GAN_legend_dict[particle]
                _ = plt.hist(reference / reference_total, bins=bins, histtype='step',
                             linewidth=2, alpha=0.6, density=True, color=color,
                             label=label, linestyle='dashed')
        flow_label = Flow_legend_dict[particle]
        if IS_v2:
            style_dict = {'label': flow_label+ ' ' + which_list[i],
                          'linestyle': 'dashdot' if which_list[i] == 'teacher' else 'solid',
                          'alpha': 1 if which_list[i] == 'student' else 0.5}
        else:
            style_dict = {'label': flow_label, 'linestyle': 'solid', 'alpha': 1}

        _ = plt.hist(plot_energy / energies, bins=bins, histtype='step', linewidth=3,
                     color=color, density='True', **style_dict)

    if use_log:
        plt.yscale('log')
        plt.xscale('log')
    if not args.no_legend:
        if IS_v2:
            entries = 2 if plot_GAN else 3
            new_leg_handles, new_leg_labels = plt.gca().get_legend_handles_labels()
        else:
            entries = 3 if plot_GAN else 2
            new_leg_handles, new_leg_labels = reorder_legend(entries, len(particles))
        plt.legend(new_leg_handles, new_leg_labels, fontsize=(16 if plot_GAN else 20), ncol=entries)

    plt.xlabel(axis_label)

    if not is_sub:

        plt.tight_layout()
        if save_it:
            plt.savefig(filename, dpi=300)
            plt.close()
        else:
            plt.show()

def plot_layer_sparsity(particle_list, which_list, layer_list, filename, layer,
                        save_it=True, plot_GAN=False, flow_thres=0., geant_thres=0., is_sub=False):
    """ plots the sparsity (number of non-zero voxel) per layer
        which_list (list of str): list of "teacher" or "student" to identify flow
        layer: str (0, 1, or 2) to indicate layer number
        save_it: bool, whether or not to save (or just show) result
    """

    assert len(particle_list) == len(layer_list)
    if not is_sub:
        plt.figure(figsize=(10, 10))
    bins = np.linspace(0, 1, 20)

    for i, particle in enumerate(particle_list):
        color = COLORS[particle]
        layer_list[i] = plthlp.thres_clip(layer_list[i], flow_thres)

        sparsity = plthlp.layer_sparsity(layer_list[i], flow_thres)

        if which_list[i] == 'teacher':
            if geant_thres not in [0., 1e-1, 1e-2, 1e-3, 1e-4]:
                raise ValueError(
                    "geant_thres {} not in pre-computed set.".format(geant_thres))
            reference_data = pd.read_hdf('plots_reference_{:1.0e}.hdf'.format(geant_thres))
            reference = reference_data.loc[particle]\
                ['sparsity_layer_{}_{:1.0e}'.format(str(layer), geant_thres)]
            label = GEANT_legend_dict[particle]
            _ = plt.hist(reference, bins=bins, histtype='stepfilled',
                         linewidth=2, alpha=0.2, density=True, color=color,
                         label=label)
            if plot_GAN:
                reference_data = pd.read_hdf('plots_reference_CaloGAN.hdf')
                reference = reference_data.loc[particle]['sparsity_layer_{}'.format(str(layer))]
                label = GAN_legend_dict[particle]
                _ = plt.hist(reference, bins=bins, histtype='step',
                             linewidth=2, alpha=0.6, density=True, color=color,
                             label=label, linestyle='dashed')

        flow_label = Flow_legend_dict[particle]
        if IS_v2:
            style_dict = {'label': flow_label+ ' ' + which_list[i],
                          'linestyle': 'dashdot' if which_list[i] == 'teacher' else 'solid',
                          'alpha': 1 if which_list[i] == 'student' else 0.5}
        else:
            style_dict = {'label': flow_label, 'linestyle': 'solid', 'alpha': 1}
        _ = plt.hist(sparsity, bins=bins, histtype='step', linewidth=3,
                     color=color, density='True', **style_dict)

    if not args.no_legend:
        if IS_v2:
            entries = 2 if plot_GAN else 3
            new_leg_handles, new_leg_labels = plt.gca().get_legend_handles_labels()
        else:
            entries = 3 if plot_GAN else 2
            new_leg_handles, new_leg_labels = reorder_legend(entries, len(particles))
        plt.legend(new_leg_handles, new_leg_labels, fontsize=(16 if plot_GAN else 20), ncol=entries)

    plt.xlabel('Sparsity in Layer {}'.format(layer))

    if not is_sub:
        plt.tight_layout()
        if save_it:
            plt.savefig(filename, dpi=300)
            plt.close()
        else:
            plt.show()

def plot_layer_E_ratio(particle_list, which_list, layer_list, filename, layer,
                       save_it=True, plot_GAN=False, flow_thres=0., geant_thres=0., is_sub=False):
    """ plots the ratio of the difference of the 2 brightest voxels to their sum
        which_list (list of str): list of "teacher" or "student" to identify flow
        layer: str (0, 1, or 2) to indicate layer number
        save_it: bool, whether or not to save (or just show) result
    """
    assert len(particle_list) == len(layer_list)

    bins = np.linspace(0, 1, 100)
    axis_label = r'''$E_{\mathrm{ratio},%(layer_id)d}$'''
    axis_label = axis_label % {'layer_id': layer}
    if not is_sub:
        plt.figure(figsize=(10, 10))

    for i, particle in enumerate(particle_list):
        color = COLORS[particle]
        layer_list[i] = plthlp.thres_clip(layer_list[i], flow_thres)
        E_ratio = plthlp.ratio_two_brightest(layer_list[i])

        if which_list[i] == 'teacher':
            if geant_thres not in [0., 1e-1, 1e-2, 1e-3, 1e-4]:
                raise ValueError(
                    "geant_thres {} not in pre-computed set.".format(geant_thres))

            reference_data = pd.read_hdf('plots_reference_{:1.0e}.hdf'.format(geant_thres))
            reference_1 = reference_data.loc[particle]\
                ['E_1_layer_{}_{:1.0e}'.format(layer, geant_thres)]
            reference_2 = reference_data.loc[particle]\
                ['E_2_layer_{}_{:1.0e}'.format(layer, geant_thres)]
            reference = ((reference_1 - reference_2) / (reference_2 + reference_1+1e-16))
            label = GEANT_legend_dict[particle]
            _ = plt.hist(reference, bins=bins, histtype='stepfilled',
                         linewidth=2, alpha=0.2, density=True, color=color,
                         label=label)
            if plot_GAN:
                reference_data = pd.read_hdf('plots_reference_CaloGAN.hdf')
                reference_1 = reference_data.loc[particle]['E_1_layer_{}'.format(layer)]
                reference_2 = reference_data.loc[particle]['E_2_layer_{}'.format(layer)]
                reference = ((reference_1 - reference_2) / (reference_2 + reference_1+1e-16))
                label = GAN_legend_dict[particle]
                _ = plt.hist(reference, bins=bins, histtype='step',
                             linewidth=2, alpha=0.6, density=True, color=color,
                             label=label, linestyle='dashed')
        flow_label = Flow_legend_dict[particle]
        if IS_v2:
            style_dict = {'label': flow_label+ ' ' + which_list[i],
                          'linestyle': 'dashdot' if which_list[i] == 'teacher' else 'solid',
                          'alpha': 1 if which_list[i] == 'student' else 0.5}
        else:
            style_dict = {'label': flow_label, 'linestyle': 'solid', 'alpha': 1}
        _ = plt.hist(E_ratio, bins=bins, histtype='step', linewidth=3,
                     color=color, density='True', **style_dict)

    if not args.no_legend:
        if IS_v2:
            entries = 2 if plot_GAN else 3
            new_leg_handles, new_leg_labels = plt.gca().get_legend_handles_labels()
        else:
            entries = 3 if plot_GAN else 2
            new_leg_handles, new_leg_labels = reorder_legend(entries, len(particles))
        plt.legend(new_leg_handles, new_leg_labels, fontsize=(16 if plot_GAN else 20), ncol=entries)

    plt.xlabel(axis_label)

    if not is_sub:
        plt.tight_layout()
        if save_it:
            plt.savefig(filename, dpi=300)
            plt.close()
        else:
            plt.show()

def plot_shower_depth(particle_list, which_list, layer_0_list, layer_1_list, layer_2_list, filename,
                      save_it=True, plot_GAN=False, flow_thres=0., geant_thres=0., is_sub=False):
    """ plots the depth of the shower, i.e. the layer that has the last energy deposition
        which_list (list of str): list of "teacher" or "student" to identify flow
        save_it: bool, whether or not to save (or just show) result
    """
    assert len(particle_list) == len(layer_0_list)
    assert len(particle_list) == len(layer_1_list)
    assert len(particle_list) == len(layer_2_list)

    bins = [0, 1, 2, 3]
    if not is_sub:
        plt.figure(figsize=(10, 10))

    for i, particle in enumerate(particle_list):
        color = COLORS[particle]
        layer_0_list[i] = plthlp.thres_clip(layer_0_list[i], flow_thres)
        layer_1_list[i] = plthlp.thres_clip(layer_1_list[i], flow_thres)
        layer_2_list[i] = plthlp.thres_clip(layer_2_list[i], flow_thres)
        assert (layer_0_list[i].shape[-1] == 288) and\
            (layer_1_list[i].shape[-1] == 144) and\
            (layer_2_list[i].shape[-1] == 72), (
                "Are you sure the input is of the right form?")
        layers_explicit = (layer_0_list[i], layer_1_list[i], layer_2_list[i])
        maxdepth = plthlp.maxdepth_nr(None, layers_explicit=layers_explicit)

        if which_list[i] == 'teacher':
            if geant_thres not in [0., 1e-1, 1e-2, 1e-3, 1e-4]:
                raise ValueError(
                    "geant_thres {} not in pre-computed set".format(geant_thres))
            reference_data = pd.read_hdf('plots_reference_{:1.0e}.hdf'.format(geant_thres))
            reference_0 = reference_data.loc[particle]\
                ['energy_layer_0_{:1.0e}'.format(geant_thres)]
            reference_1 = reference_data.loc[particle]\
                ['energy_layer_1_{:1.0e}'.format(geant_thres)]
            reference_2 = reference_data.loc[particle]\
                ['energy_layer_2_{:1.0e}'.format(geant_thres)]

            layers_explicit_ref = (reference_0, reference_1, reference_2)
            ref_depth = plthlp.maxdepth_nr(None, layers_explicit=layers_explicit_ref)
            label = GEANT_legend_dict[particle]
            _ = plt.hist(ref_depth, bins=bins, histtype='stepfilled',
                         linewidth=2, alpha=0.2, density=True, color=color,
                         label=label)
            if plot_GAN:
                reference_data = pd.read_hdf('plots_reference_CaloGAN.hdf')
                reference_0 = reference_data.loc[particle]['energy_layer_0']
                reference_1 = reference_data.loc[particle]['energy_layer_1']
                reference_2 = reference_data.loc[particle]['energy_layer_2']

                layers_explicit_ref = (reference_0, reference_1, reference_2)
                ref_depth = plthlp.maxdepth_nr(None, layers_explicit=layers_explicit_ref)
                label = GAN_legend_dict[particle]
                _ = plt.hist(ref_depth, bins=bins, histtype='step',
                             linewidth=2, alpha=0.6, density=True, color=color,
                             label=label, linestyle='dashed')

        flow_label = Flow_legend_dict[particle]
        if IS_v2:
            style_dict = {'label': flow_label+ ' ' + which_list[i],
                          'linestyle': 'dashdot' if which_list[i] == 'teacher' else 'solid',
                          'alpha': 1 if which_list[i] == 'student' else 0.5}
        else:
            style_dict = {'label': flow_label, 'linestyle': 'solid', 'alpha': 1}
        _ = plt.hist(maxdepth, bins=bins, histtype='step', linewidth=3,
                     color=color, density='True', **style_dict)

    if not args.no_legend:
        if IS_v2:
            entries = 2 if plot_GAN else 3
            new_leg_handles, new_leg_labels = plt.gca().get_legend_handles_labels()
        else:
            entries = 3 if plot_GAN else 2
            new_leg_handles, new_leg_labels = reorder_legend(entries, len(particles))
        plt.legend(new_leg_handles, new_leg_labels, fontsize=(16 if plot_GAN else 20), ncol=entries,
                   loc='upper left')
    plt.xlabel(r'Max Depth $d$ (layer)')

    if not is_sub:
        plt.tight_layout()
        if save_it:
            plt.savefig(filename, dpi=300)
            plt.close()
        else:
            plt.show()

def plot_depth_weighted_total_energy(particle_list, which_list, layer_0_list, layer_1_list,
                                     layer_2_list, filename, save_it=True, plot_GAN=False,
                                     flow_thres=0., geant_thres=0., is_sub=False):
    """ plots the depth-weighted total energy deposit in all 3 layers for a given batch
        which_list (list of str): list of "teacher" or "student" to identify flow
        filename: file path to save file
        save_it: bool, whether or not to save (or just show) result
    """
    assert len(particle_list) == len(layer_0_list)
    assert len(particle_list) == len(layer_1_list)
    assert len(particle_list) == len(layer_2_list)

    bins = np.logspace(1, 5, 100)
    if not is_sub:
        plt.figure(figsize=(10, 10))

    for i, particle in enumerate(particle_list):
        color = COLORS[particle]
        layer_0_list[i] = plthlp.thres_clip(layer_0_list[i], flow_thres)
        layer_1_list[i] = plthlp.thres_clip(layer_1_list[i], flow_thres)
        layer_2_list[i] = plthlp.thres_clip(layer_2_list[i], flow_thres)
        assert (layer_0_list[i].shape[-1] == 288) and\
            (layer_1_list[i].shape[-1] == 144) and\
            (layer_2_list[i].shape[-1] == 72), (
                "Are you sure the input is of the right form?")

        layers_explicit = (layer_0_list[i], layer_1_list[i], layer_2_list[i])
        energies = plthlp.depth_weighted_energy(None, layers_explicit=layers_explicit)

        if which_list[i] == 'teacher':
            if geant_thres not in [0., 1e-1, 1e-2, 1e-3, 1e-4]:
                raise ValueError(
                    "geant_thres {} not in pre-computed set".format(geant_thres))
            reference_data = pd.read_hdf('plots_reference_{:1.0e}.hdf'.format(geant_thres))
            reference = reference_data.loc[particle]['energy_layer_1_{:1.0e}'.format(geant_thres)]
            reference = reference + 2.*reference_data.loc[particle]\
                ['energy_layer_2_{:1.0e}'.format(geant_thres)]
            label = GEANT_legend_dict[particle]
            _ = plt.hist(reference, bins=bins, histtype='stepfilled',
                         linewidth=2, alpha=0.2, density=True, color=color,
                         label=label)
            if plot_GAN:
                reference_data = pd.read_hdf('plots_reference_CaloGAN.hdf')
                reference = reference_data.loc[particle]['energy_layer_1']
                reference = reference + 2.*reference_data.loc[particle]['energy_layer_2']
                label = GAN_legend_dict[particle]
                _ = plt.hist(reference, bins=bins, histtype='step',
                             linewidth=2, alpha=0.6, density=True, color=color,
                             label=label, linestyle='dashed')

        flow_label = Flow_legend_dict[particle]
        if IS_v2:
            style_dict = {'label': flow_label+ ' ' + which_list[i],
                          'linestyle': 'dashdot' if which_list[i] == 'teacher' else 'solid',
                          'alpha': 1 if which_list[i] == 'student' else 0.5}
        else:
            style_dict = {'label': flow_label, 'linestyle': 'solid', 'alpha': 1}
        _ = plt.hist(energies, bins=bins, histtype='step', linewidth=3,
                     color=color, density='True', **style_dict)

    if not args.no_legend:
        if IS_v2:
            entries = 2 if plot_GAN else 3
            new_leg_handles, new_leg_labels = plt.gca().get_legend_handles_labels()
        else:
            entries = 3 if plot_GAN else 2
            new_leg_handles, new_leg_labels = reorder_legend(entries, len(particles))
        plt.legend(new_leg_handles, new_leg_labels, fontsize=(16 if plot_GAN else 20), ncol=entries)

    plt.ylim([1e-8, 3e-2])
    plt.yscale('log')
    plt.xscale('log')

    plt.xlabel(r'Depth-weighted total energy $l_d$')

    if not is_sub:

        plt.tight_layout()
        if save_it:
            plt.savefig(filename, dpi=300)
            plt.close()
        else:
            plt.show()

def plot_layer_lateral_width(particle_list, which_list, layer_list, filename, layer, save_it=True,
                             plot_GAN=False, flow_thres=0., geant_thres=0., is_sub=False):
    """ plots the standard deviation of the transverse energy profile per layer,
        in units of cell numbers for a given batch
        which_list (list of str): list of "teacher" or "student" to identify flow
        filename: file path to save file
        layer: str (0, 1, or 2) to indicate layer number
        save_it: bool, whether or not to save (or just show) result
    """
    assert len(particle_list) == len(layer_list)
    axis_label = r'''$\sigma_{%(layer_id)d}$'''
    axis_label = axis_label % {'layer_id': layer}

    if layer == 0:
        bins = np.logspace(0, 3, 100)
        x_max = 200
        y_max = 5
    elif layer == 1:
        bins = np.logspace(0, 2, 100)
        x_max = 200
        y_max = 2
    elif layer == 2:
        bins = np.logspace(0, 3, 100)
        x_max = 300
        y_max = 2

    if not is_sub:
        plt.figure(figsize=(10, 10))

    for i, particle in enumerate(particle_list):
        color = COLORS[particle]
        layer_list[i] = plthlp.thres_clip(layer_list[i], flow_thres)

        dist = plthlp.center_of_energy_std(layer_list[i], layer, 'phi')

        if which_list[i] == 'teacher':
            if geant_thres not in [0., 1e-1, 1e-2, 1e-3, 1e-4]:
                raise ValueError(
                    "geant_thres {} not in pre-computed set.".format(geant_thres))
            reference_data = pd.read_hdf('plots_reference_{:1.0e}.hdf'.format(geant_thres))
            reference = reference_data.loc[particle]\
                ['layer_{}_lateral_width_{:1.0e}'.format(str(layer), geant_thres)]
            label = GEANT_legend_dict[particle]
            _ = plt.hist(reference, bins=bins, histtype='stepfilled',
                         linewidth=2, alpha=0.2, density=True, color=color,
                         label=label)
            if plot_GAN:
                reference_data = pd.read_hdf('plots_reference_CaloGAN.hdf')
                reference = reference_data.loc[particle]\
                    ['layer_{}_lateral_width'.format(str(layer))]
                label = GAN_legend_dict[particle]
                _ = plt.hist(reference, bins=bins, histtype='step',
                             linewidth=2, alpha=0.6, density=True, color=color,
                             label=label, linestyle='dashed')

        flow_label = Flow_legend_dict[particle]
        if IS_v2:
            style_dict = {'label': flow_label+ ' ' + which_list[i],
                          'linestyle': 'dashdot' if which_list[i] == 'teacher' else 'solid',
                          'alpha': 1 if which_list[i] == 'student' else 0.5}
        else:
            style_dict = {'label': flow_label, 'linestyle': 'solid', 'alpha': 1}
        _ = plt.hist(dist, bins=bins, histtype='step', linewidth=3,
                     color=color, density='True', **style_dict)

    plt.yscale('log')
    plt.xscale('log')
    plt.xlim(xmax=x_max)
    plt.ylim(ymax=y_max)
    if not args.no_legend:
        if IS_v2:
            entries = 2 if plot_GAN else 3
            new_leg_handles, new_leg_labels = plt.gca().get_legend_handles_labels()
        else:
            entries = 3 if plot_GAN else 2
            new_leg_handles, new_leg_labels = reorder_legend(entries, len(particles))
        plt.legend(new_leg_handles, new_leg_labels, fontsize=(16 if plot_GAN else 20), ncol=entries)

    plt.xlabel(axis_label)
    if not is_sub:

        plt.tight_layout()
        if save_it:
            plt.savefig(filename, dpi=300)
            plt.close()
        else:
            plt.show()

def plot_depth_weighted_energy_normed(particle_list, which_list, layer_0_list, layer_1_list,
                                      layer_2_list, filename, save_it=True, plot_GAN=False,
                                      flow_thres=0., geant_thres=0., is_sub=False):
    """ plots the depth-weighted total energy deposit in all 3 layers
        normalized by the total deposited energy for a given batch
        which_list (list of str): list of "teacher" or "student" to identify flow
        filename: file path to save file
        save_it: bool, whether or not to save (or just show) result
    """
    assert len(particle_list) == len(layer_0_list)
    assert len(particle_list) == len(layer_1_list)
    assert len(particle_list) == len(layer_2_list)

    bins = np.linspace(0.4, 2., 100)
    if not is_sub:
        plt.figure(figsize=(10, 10))

    for i, particle in enumerate(particle_list):
        color = COLORS[particle]
        layer_0_list[i] = plthlp.thres_clip(layer_0_list[i], flow_thres)
        layer_1_list[i] = plthlp.thres_clip(layer_1_list[i], flow_thres)
        layer_2_list[i] = plthlp.thres_clip(layer_2_list[i], flow_thres)
        assert (layer_0_list[i].shape[-1] == 288) and\
            (layer_1_list[i].shape[-1] == 144) and\
            (layer_2_list[i].shape[-1] == 72), (
                "Are you sure the input is of the right form?")
        layers_explicit = (layer_0_list[i], layer_1_list[i], layer_2_list[i])
        energies = plthlp.depth_weighted_energy(None, layers_explicit=layers_explicit) / \
            (plthlp.energy_sum(layer_0_list[i], normalization=1.) + \
             plthlp.energy_sum(layer_1_list[i], normalization=1.) + \
             plthlp.energy_sum(layer_2_list[i], normalization=1.))

        if which_list[i] == 'teacher':
            if geant_thres not in [0., 1e-1, 1e-2, 1e-3, 1e-4]:
                raise ValueError(
                    "geant_thres {} not in pre-computed set".format(geant_thres))

            reference_data = pd.read_hdf('plots_reference_{:1.0e}.hdf'.format(geant_thres))
            reference = reference_data.loc[particle]['energy_layer_1_{:1.0e}'.format(geant_thres)]
            reference = reference + 2.*reference_data.loc[particle]\
                ['energy_layer_2_{:1.0e}'.format(geant_thres)]
            reference = reference / (reference_data.loc[particle]\
                          ['energy_layer_0_{:1.0e}'.format(geant_thres)]+
                                     reference_data.loc[particle]\
                          ['energy_layer_1_{:1.0e}'.format(geant_thres)]+
                                     reference_data.loc[particle]\
                          ['energy_layer_2_{:1.0e}'.format(geant_thres)])
            label = GEANT_legend_dict[particle]
            _ = plt.hist(reference, bins=bins, histtype='stepfilled',
                         linewidth=2, alpha=0.2, density=True, color=color,
                         label=label)
            if plot_GAN:
                reference_data = pd.read_hdf('plots_reference_CaloGAN.hdf')
                reference = reference_data.loc[particle]['energy_layer_1']
                reference = reference + 2.*reference_data.loc[particle]['energy_layer_2']
                reference = reference / (reference_data.loc[particle]['energy_layer_0']+
                                         reference_data.loc[particle]['energy_layer_1']+
                                         reference_data.loc[particle]['energy_layer_2'])
                label = GAN_legend_dict[particle]
                _ = plt.hist(reference, bins=bins, histtype='step',
                             linewidth=2, alpha=0.6, density=True, color=color,
                             label=label, linestyle='dashed')

        flow_label = Flow_legend_dict[particle]
        if IS_v2:
            style_dict = {'label': flow_label+ ' ' + which_list[i],
                          'linestyle': 'dashdot' if which_list[i] == 'teacher' else 'solid',
                          'alpha': 1 if which_list[i] == 'student' else 0.5}
        else:
            style_dict = {'label': flow_label, 'linestyle': 'solid', 'alpha': 1}
        _ = plt.hist(energies, bins=bins, histtype='step', linewidth=3,
                     color=color, density='True', **style_dict)

    if not args.no_legend:
        if IS_v2:
            entries = 2 if plot_GAN else 3
            new_leg_handles, new_leg_labels = plt.gca().get_legend_handles_labels()
        else:
            entries = 3 if plot_GAN else 2
            new_leg_handles, new_leg_labels = reorder_legend(entries, len(particles))
        plt.legend(new_leg_handles, new_leg_labels, fontsize=(16 if plot_GAN else 20), ncol=entries)

    plt.ylim([0., 7.])

    plt.xlabel(r'Shower Depth $s_d$')
    if not is_sub:

        plt.tight_layout()
        if save_it:
            plt.savefig(filename, dpi=300)
            plt.close()
        else:
            plt.show()

def plot_depth_weighted_energy_normed_std(particle_list, which_list, layer_0_list, layer_1_list,
                                          layer_2_list, filename, save_it=True, plot_GAN=False,
                                          flow_thres=0., geant_thres=0., is_sub=False):
    """ plots the standard deviation of the depth-weighted total energy deposit in all 3 layers
        normalized by the total deposited energy for a given batch
        which_list (list of str): list of "teacher" or "student" to identify flow
        filename: file path to save file
        save_it: bool, whether or not to save (or just show) result
    """
    assert len(particle_list) == len(layer_0_list)
    assert len(particle_list) == len(layer_1_list)
    assert len(particle_list) == len(layer_2_list)

    bins = np.linspace(0., 0.9, 100)
    if not is_sub:
        plt.figure(figsize=(10, 10))

    for i, particle in enumerate(particle_list):
        color = COLORS[particle]
        layer_0_list[i] = plthlp.thres_clip(layer_0_list[i], flow_thres)
        layer_1_list[i] = plthlp.thres_clip(layer_1_list[i], flow_thres)
        layer_2_list[i] = plthlp.thres_clip(layer_2_list[i], flow_thres)

        assert (layer_0_list[i].shape[-1] == 288) and\
            (layer_1_list[i].shape[-1] == 144) and\
            (layer_2_list[i].shape[-1] == 72), (
                "Are you sure the input is of the right form?")
        energies = plthlp.energy_sum(layer_0_list[i], normalization=1.) + \
             (layer1 := plthlp.energy_sum(layer_1_list[i], normalization=1.)) + \
             (layer2 := plthlp.energy_sum(layer_2_list[i], normalization=1.))
        if which_list[i] == 'teacher':
            if geant_thres not in [0., 1e-1, 1e-2, 1e-3, 1e-4]:
                raise ValueError(
                    "geant_thres {} not in pre-computed set".format(geant_thres))
            reference_data = pd.read_hdf('plots_reference_{:1.0e}.hdf'.format(geant_thres))
            reference1 = reference_data.loc[particle]\
                ['energy_layer_1_{:1.0e}'.format(geant_thres)]
            reference2 = reference_data.loc[particle]\
                ['energy_layer_2_{:1.0e}'.format(geant_thres)]
            referencetot = reference1 + reference2 + reference_data.loc[particle]\
                ['energy_layer_0_{:1.0e}'.format(geant_thres)]
            label = GEANT_legend_dict[particle]
            _ = plt.hist(plthlp.layer_std(reference1, reference2, referencetot), bins=bins,
                         histtype='stepfilled',
                         linewidth=2, alpha=0.2, density=True, color=color,
                         label=label)
            if plot_GAN:
                reference_data = pd.read_hdf('plots_reference_CaloGAN.hdf')
                reference1 = reference_data.loc[particle]['energy_layer_1']
                reference2 = reference_data.loc[particle]['energy_layer_2']
                referencetot = reference1 + reference2 + reference_data.loc[particle]\
                    ['energy_layer_0']
                label = GAN_legend_dict[particle]
                _ = plt.hist(plthlp.layer_std(reference1, reference2, referencetot), bins=bins,
                             histtype='step',
                             linewidth=2, alpha=0.6, density=True, color=color,
                             label=label, linestyle='dashed')

        flow_label = Flow_legend_dict[particle]
        if IS_v2:
            style_dict = {'label': flow_label+ ' ' + which_list[i],
                          'linestyle': 'dashdot' if which_list[i] == 'teacher' else 'solid',
                          'alpha': 1 if which_list[i] == 'student' else 0.5}
        else:
            style_dict = {'label': flow_label, 'linestyle': 'solid', 'alpha': 1}
        _ = plt.hist(plthlp.layer_std(layer1, layer2, energies), bins=bins, histtype='step',
                     linewidth=3, color=color, density='True', **style_dict)
    plt.ylim([0., 7.])

    if IS_v2:
        entries = 2 if plot_GAN else 3
        new_leg_handles, new_leg_labels = plt.gca().get_legend_handles_labels()
    else:
        entries = 3 if plot_GAN else 2
        new_leg_handles, new_leg_labels = reorder_legend(entries, len(particles))
    if not args.no_legend:
        plt.legend(new_leg_handles, new_leg_labels, fontsize=(16 if plot_GAN else 20), ncol=entries)

    plt.xlabel(r'Shower Depth Width $\sigma_{s_d}$')

    if not is_sub:
        plt.tight_layout()
        if save_it:
            plt.savefig(filename, dpi=300)
            plt.close()
        else:
            plt.show()
    if args.no_legend:
        return new_leg_handles, new_leg_labels, entries

def plot_legend(handles, labels, entries, filename, save_it=True):
    """ plots the legend of the plot(s) before"""
    fig_leg = plt.figure(figsize=(8., 2./3.)) # if 1 particle, use (8,2) for 3 particles
    ax_leg = fig_leg.add_subplot(111)
    # add the legend from the previous axes
    ax_leg.legend(handles, labels, fontsize=(16 if entries == 3 else 20),
                  ncol=entries, loc='center')
    # hide the axes frame and the x/y labels
    ax_leg.axis('off')
    if save_it:
        plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()
    else:
        plt.show()

def plot_brightest_voxel(particle_list, which_list, layer_list, filename, layer,
                         which_voxel=1, save_it=True, plot_GAN=False,
                         flow_thres=0., geant_thres=0., is_sub=False):
    """ plots the ratio of the which_voxel brightest voxels to the energy
        deposited in the layer
        which_list (list of str): list of "teacher" or "student" to identify flow
        filename: file path to save file
        layer: str (0, 1, or 2) to indicate layer number
        save_it: bool, whether or not to save (or just show) result
    """
    assert len(particle_list) == len(layer_list)

    bins = np.linspace(0, 1./which_voxel, 100)

    label = r'''$E_{\mathrm{{%(which_id)d}. brightest}, \mathrm{layer } %(layer_id)d}$'''
    axis_label = label % {'which_id': which_voxel, 'layer_id': layer}

    if not is_sub:
        plt.figure(figsize=(10, 10))

    for i, particle in enumerate(particle_list):
        color = COLORS[particle]
        layer_list[i] = plthlp.thres_clip(layer_list[i], flow_thres)

        ratio = plthlp.n_brightest_voxel(layer_list[i], [which_voxel])
        if which_list[i] == 'teacher':
            if geant_thres not in [0., 1e-1, 1e-2, 1e-3, 1e-4]:
                raise ValueError(
                    "geant_thres {} not in pre-computed set.".format(geant_thres))
            reference_data = pd.read_hdf('plots_reference_{:1.0e}.hdf'.format(geant_thres))
            reference = reference_data.loc[particle]\
                ['E_{}_layer_{}_{:1.0e}'.format(which_voxel, layer, geant_thres)]
            reference_tot = reference_data.loc[particle]\
                ['energy_layer_{}_{:1.0e}'.format(layer, geant_thres)]
            reference = (reference/(reference_tot+ 1e-16))

            label = GEANT_legend_dict[particle]
            _ = plt.hist(reference, bins=bins, histtype='stepfilled',
                         linewidth=2, alpha=0.2, density=True, color=color,
                         label=label)
            if plot_GAN:
                reference_data = pd.read_hdf('plots_reference_CaloGAN.hdf')
                reference = reference_data.loc[particle]['E_{}_layer_{}'.format(which_voxel, layer)]
                reference_tot = reference_data.loc[particle]['energy_layer_{}'.format(layer)]
                reference = (reference/(reference_tot+ 1e-16))
                label = GAN_legend_dict[particle]
                _ = plt.hist(reference, bins=bins, histtype='step',
                             linewidth=2, alpha=0.6, density=True, color=color,
                             label=label, linestyle='dashed')

        flow_label = Flow_legend_dict[particle]
        if IS_v2:
            style_dict = {'label': flow_label+ ' ' + which_list[i],
                          'linestyle': 'dashdot' if which_list[i] == 'teacher' else 'solid',
                          'alpha': 1 if which_list[i] == 'student' else 0.5}
        else:
            style_dict = {'label': flow_label, 'linestyle': 'solid', 'alpha': 1}
        _ = plt.hist(ratio, bins=bins, histtype='step', linewidth=3,
                     color=color, density='True', **style_dict)

    if not args.no_legend:
        if IS_v2:
            entries = 2 if plot_GAN else 3
            new_leg_handles, new_leg_labels = plt.gca().get_legend_handles_labels()
        else:
            entries = 3 if plot_GAN else 2
            new_leg_handles, new_leg_labels = reorder_legend(entries, len(particles))
        plt.legend(new_leg_handles, new_leg_labels, fontsize=(16 if plot_GAN else 20), ncol=entries)

    plt.xlabel(axis_label)

    if not is_sub:
        plt.tight_layout()
        if save_it:
            plt.savefig(filename, dpi=300)
            plt.close()
        else:
            plt.show()

def plot_centroid_correlation(particle_list, which_list, layer1, layer1_list, layer2, layer2_list,
                              filename, scan='phi', save_it=True,
                              plot_GAN=False, flow_thres=0., geant_thres=0., is_sub=False):
    """ plots the difference between the x positions of the centroid of layer1 and layer2
        which_list (list of str): list of "teacher" or "student" to identify flow
        filename: file path to save file
        save_it: bool, whether or not to save (or just show) result
    """
    assert len(particle_list) == len(layer1_list)
    assert len(particle_list) == len(layer2_list)
    if layer2 < layer1:
        layer1, layer2 = layer2, layer1
        layer1_list, layer2_list = layer2_list, layer1_list

    label = r'''$\langle \%(scan)s_{%(layer1)d}\rangle - \langle \%(scan)s_{%(layer2)d}\rangle$'''
    axis_label = label % {'scan': scan, 'layer1': layer1, 'layer2': layer2}

    bins = np.linspace(-120, 120, 50)
    if not is_sub:
        plt.figure(figsize=(10, 10))

    for i, particle in enumerate(particle_list):
        color = COLORS[particle]
        layer1_list[i] = plthlp.thres_clip(layer1_list[i], flow_thres)
        layer2_list[i] = plthlp.thres_clip(layer2_list[i], flow_thres)

        x1 = plthlp.center_of_energy(layer1_list[i], layer1, scan)
        x2 = plthlp.center_of_energy(layer2_list[i], layer2, scan)

        dist = x1-x2
        if which_list[i] == 'teacher':
            if geant_thres not in [0., 1e-1, 1e-2, 1e-3, 1e-4]:
                raise ValueError(
                    "geant_thres {} not in pre-computed set.".format(geant_thres))

            reference_data = pd.read_hdf('plots_reference_{:1.0e}.hdf'.format(geant_thres))
            reference1 = reference_data.loc[particle]\
                ['layer_{}_centroid_{}_{:1.0e}'.format(str(layer1), str(scan), geant_thres)]
            reference2 = reference_data.loc[particle]\
                ['layer_{}_centroid_{}_{:1.0e}'.format(str(layer2), str(scan), geant_thres)]
            label = GEANT_legend_dict[particle]
            _ = plt.hist(reference1-reference2, bins=bins, histtype='stepfilled',
                         linewidth=2, alpha=0.2, density=True, color=color,
                         label=label)
            if plot_GAN:
                reference_data = pd.read_hdf('plots_reference_CaloGAN.hdf')
                reference1 = reference_data.loc[particle]\
                    ['layer_{}_centroid_{}'.format(str(layer1), str(scan))]
                reference2 = reference_data.loc[particle]\
                    ['layer_{}_centroid_{}'.format(str(layer2), str(scan))]
                label = GAN_legend_dict[particle]
                _ = plt.hist(reference1-reference2, bins=bins, histtype='step',
                             linewidth=2, alpha=0.6, density=True, color=color,
                             label=label, linestyle='dashed')

        flow_label = Flow_legend_dict[particle]
        if IS_v2:
            style_dict = {'label': flow_label+ ' ' + which_list[i],
                          'linestyle': 'dashdot' if which_list[i] == 'teacher' else 'solid',
                          'alpha': 1 if which_list[i] == 'student' else 0.5}
        else:
            style_dict = {'label': flow_label, 'linestyle': 'solid', 'alpha': 1}
        _ = plt.hist(dist, bins=bins, histtype='step', linewidth=3,
                     color=color, density='True', **style_dict)

    if not args.no_legend:
        if IS_v2:
            entries = 2 if plot_GAN else 3
            new_leg_handles, new_leg_labels = plt.gca().get_legend_handles_labels()
        else:
            entries = 3 if plot_GAN else 2
            new_leg_handles, new_leg_labels = reorder_legend(entries, len(particles))
        plt.legend(new_leg_handles, new_leg_labels, fontsize=(16 if plot_GAN else 20), ncol=entries)

    plt.yscale('log')
    plt.xlabel(axis_label)

    if not is_sub:
        plt.tight_layout()
        if save_it:
            plt.savefig(filename, dpi=300)
            plt.close()
        else:
            plt.show()

def plot_centroid(particle_list, which_list, layer1, layer1_list, filename, scan='phi', save_it=True,
                  plot_GAN=False, flow_thres=0., geant_thres=0., is_sub=False):
    """ plots the centroid of the given layer
        which_list (list of str): list of "teacher" or "student" to identify flow
        filename: file path to save file
        save_it: bool, whether or not to save (or just show) result
    """
    assert len(particle_list) == len(layer1_list)

    label = r'''$\langle \%(scan)s_{%(layer1)d}\rangle $'''
    axis_label = label % {'scan': scan, 'layer1': layer1}

    bins = np.linspace(-120, 120, 50)
    if not is_sub:
        plt.figure(figsize=(10, 10))

    for i, particle in enumerate(particle_list):
        color = COLORS[particle]
        layer1_list[i] = plthlp.thres_clip(layer1_list[i], flow_thres)

        dist = plthlp.center_of_energy(layer1_list[i], layer1, scan)

        if which_list[i] == 'teacher':
            if geant_thres not in [0., 1e-1, 1e-2, 1e-3, 1e-4]:
                raise ValueError(
                    "geant_thres {} not in pre-computed set.".format(geant_thres))

            reference_data = pd.read_hdf('plots_reference_{:1.0e}.hdf'.format(geant_thres))
            reference1 = reference_data.loc[particle]\
                ['layer_{}_centroid_{}_{:1.0e}'.format(str(layer1), str(scan), geant_thres)]
            label = GEANT_legend_dict[particle]
            _ = plt.hist(reference1, bins=bins, histtype='stepfilled',
                         linewidth=2, alpha=0.2, density=True, color=color,
                         label=label)
            if plot_GAN:
                reference_data = pd.read_hdf('plots_reference_CaloGAN.hdf')
                reference1 = reference_data.loc[particle]\
                    ['layer_{}_centroid_{}'.format(str(layer1), str(scan))]
                label = GAN_legend_dict[particle]
                _ = plt.hist(reference1, bins=bins, histtype='step',
                             linewidth=2, alpha=0.6, density=True, color=color,
                             label=label, linestyle='dashed')

        flow_label = Flow_legend_dict[particle]
        if IS_v2:
            style_dict = {'label': flow_label+ ' ' + which_list[i],
                          'linestyle': 'dashdot' if which_list[i] == 'teacher' else 'solid',
                          'alpha': 1 if which_list[i] == 'student' else 0.5}
        else:
            style_dict = {'label': flow_label, 'linestyle': 'solid', 'alpha': 1}
        _ = plt.hist(dist, bins=bins, histtype='step', linewidth=3,
                     color=color, density='True', **style_dict)

    if not args.no_legend:
        if IS_v2:
            entries = 2 if plot_GAN else 3
            new_leg_handles, new_leg_labels = plt.gca().get_legend_handles_labels()
        else:
            entries = 3 if plot_GAN else 2
            new_leg_handles, new_leg_labels = reorder_legend(entries, len(particles))
        plt.legend(new_leg_handles, new_leg_labels, fontsize=(16 if plot_GAN else 20), ncol=entries)

    plt.yscale('log')
    plt.xlabel(axis_label)

    if not is_sub:
        plt.tight_layout()
        if save_it:
            plt.savefig(filename, dpi=300)
            plt.close()
        else:
            plt.show()

def read_hdf5(arg, particle_type):
    """ extracts layers and energy from hdf5 """
    h5py_file = h5py.File(eval('arg.'+particle_type+'_file'), 'r')
    if arg.num_events is None:
        data_layer_0 = h5py_file['layer_0'][:]
        data_layer_1 = h5py_file['layer_1'][:]
        data_layer_2 = h5py_file['layer_2'][:]
        data_energy = h5py_file['energy'][:]
    else:
        num = int(arg.num_events)
        data_layer_0 = h5py_file['layer_0'][:num]
        data_layer_1 = h5py_file['layer_1'][:num]
        data_layer_2 = h5py_file['layer_2'][:num]
        data_energy = h5py['energy'][:num]
    h5py_file.close()
    return data_layer_0, data_layer_1, data_layer_2, data_energy

def reorder_legend(num_entries, num_particles):
    """ reorders the entries for the legend """
    leg_handles, leg_labels = plt.gca().get_legend_handles_labels()
    new_order = np.arange(num_entries*num_particles).reshape(num_particles, num_entries).T.flatten()
    new_labels = [leg_labels[k] for k in new_order]
    new_handles = [leg_handles[k] for k in new_order]
    return new_handles, new_labels


if __name__ == '__main__':
    args = parser.parse_args()

    if (args.eplus_file is None) and (args.gamma_file is None) and (args.piplus_file is None):
        raise ValueError("Need at least one of the 3 CaloFlow files!")

    particles = []
    flow_type = []
    layer_0 = []
    layer_1 = []
    layer_2 = []
    is_v2 = False
    if args.eplus_file is not None:
        particles.append('eplus')
        flow_type.append('teacher')
        eplus_layer_0, eplus_layer_1, eplus_layer_2, eplus_energy = read_hdf5(args, 'eplus')
        layer_0.append(eplus_layer_0.reshape(-1, INPUT_SIZE['0']))
        layer_1.append(eplus_layer_1.reshape(-1, INPUT_SIZE['1']))
        layer_2.append(eplus_layer_2.reshape(-1, INPUT_SIZE['2']))
    if args.gamma_file is not None:
        particles.append('gamma')
        flow_type.append('teacher')
        gamma_h5py = h5py.File(args.gamma_file, 'r')
        gamma_layer_0, gamma_layer_1, gamma_layer_2, gamma_energy = read_hdf5(args, 'gamma')
        layer_0.append(gamma_layer_0.reshape(-1, INPUT_SIZE['0']))
        layer_1.append(gamma_layer_1.reshape(-1, INPUT_SIZE['1']))
        layer_2.append(gamma_layer_2.reshape(-1, INPUT_SIZE['2']))
    if args.piplus_file is not None:
        particles.append('piplus')
        flow_type.append('teacher')
        piplus_layer_0, piplus_layer_1, piplus_layer_2, piplus_energy = read_hdf5(args, 'piplus')
        layer_0.append(piplus_layer_0.reshape(-1, INPUT_SIZE['0']))
        layer_1.append(piplus_layer_1.reshape(-1, INPUT_SIZE['1']))
        layer_2.append(piplus_layer_2.reshape(-1, INPUT_SIZE['2']))
    if args.eplus_student_file is not None:
        IS_v2 = True
        particles.append('eplus')
        flow_type.append('student')
        eplus_layer_0, eplus_layer_1, eplus_layer_2, eplus_energy = read_hdf5(args, 'eplus_student')
        layer_0.append(eplus_layer_0.reshape(-1, INPUT_SIZE['0']))
        layer_1.append(eplus_layer_1.reshape(-1, INPUT_SIZE['1']))
        layer_2.append(eplus_layer_2.reshape(-1, INPUT_SIZE['2']))
    if args.gamma_student_file is not None:
        IS_v2 = True
        particles.append('gamma')
        flow_type.append('student')
        gamma_h5py = h5py.File(args.gamma_file, 'r')
        gamma_layer_0, gamma_layer_1, gamma_layer_2, gamma_energy = read_hdf5(args, 'gamma_student')
        layer_0.append(gamma_layer_0.reshape(-1, INPUT_SIZE['0']))
        layer_1.append(gamma_layer_1.reshape(-1, INPUT_SIZE['1']))
        layer_2.append(gamma_layer_2.reshape(-1, INPUT_SIZE['2']))
    if args.piplus_student_file is not None:
        IS_v2 = True
        particles.append('piplus')
        flow_type.append('student')
        piplus_layer_0, piplus_layer_1, piplus_layer_2, piplus_energy = read_hdf5(args,
                                                                                  'piplus_student')
        layer_0.append(piplus_layer_0.reshape(-1, INPUT_SIZE['0']))
        layer_1.append(piplus_layer_1.reshape(-1, INPUT_SIZE['1']))
        layer_2.append(piplus_layer_2.reshape(-1, INPUT_SIZE['2']))


    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir)

    print("Plotting histograms for " + str(particles) + " in folder " + args.results_dir)
    #plotting single files:
    matplotlib.rcParams.update({'font.size': 32})

    ### plotting single histograms (not used for paper):
    #
    #filename = 'generated_samples_E_total'+('_GAN' if args.plot_GAN else '')+'.png'
    #plot_total_energy(particles, flow_type, layer_0, layer_1, layer_2,
    #                  os.path.join(args.results_dir, filename),
    #                  save_it=not args.show, plot_GAN=args.plot_GAN, flow_thres=0., geant_thres=0.)
    #
    #for layer_id, layer_content in enumerate([layer_0, layer_1, layer_2]):
    #    filename = 'generated_samples_E_layer_'+str(layer_id)+('_GAN' if args.plot_GAN else '')+\
    #        '.png'
    #    plot_layer_energy(particles, flow_type, layer_content,
    #                      os.path.join(args.results_dir, filename),
    #                      layer_id, save_it=not args.show, plot_GAN=args.plot_GAN,
    #                      flow_thres=0., geant_thres=0.)
    #
    #for layer_id in [0, 1, 2]:
    #    for use_log in [False, True]:
    #        filename = 'energy_fraction_layer_'+str(layer_id)+ (use_log)*'_log'+\
    #            ('_GAN' if args.plot_GAN else '')+'.pdf'
    #        plot_energy_fraction(particles, flow_type, layer_0, layer_1, layer_2,
    #                             os.path.join(args.results_dir, filename), layer_id,
    #                             save_it=not args.show, plot_GAN=args.plot_GAN, use_log=use_log,
    #                             flow_thres=0., geant_thres=0.)

    #for layer_id, layer_content in enumerate([layer_0, layer_1, layer_2]):
    #    filename = 'sparsity_layer_'+str(layer_id) +('_GAN' if args.plot_GAN else '')+'.pdf'
    #    plot_layer_sparsity(particles, flow_type, layer_content,
    #                        os.path.join(args.results_dir, filename),
    #                        layer_id, save_it=not args.show, plot_GAN=args.plot_GAN,
    #                        flow_thres=0., geant_thres=0.)
    #
    #for layer_id, layer_content in enumerate([layer_0, layer_1, layer_2]):
    #    filename = 'e_ratio_layer_'+str(layer_id) +('_GAN' if args.plot_GAN else '')+'.pdf'
    #    plot_layer_E_ratio(particles, flow_type, layer_content,
    #                       os.path.join(args.results_dir, filename),
    #                       layer_id, save_it=not args.show, plot_GAN=args.plot_GAN,
    #                       flow_thres=0., geant_thres=0.)

    #filename = 'shower_depth'+('_GAN' if args.plot_GAN else '')+'.png'
    #plot_shower_depth(particles, flow_type, layer_0, layer_1, layer_2,
    #                  os.path.join(args.results_dir, filename),
    #                  save_it=not args.show, plot_GAN=args.plot_GAN, flow_thres=0., geant_thres=0.)

    #filename = 'depth_weighted_total_energy'+('_GAN' if args.plot_GAN else '')+'.pdf'
    #plot_depth_weighted_total_energy(particles, flow_type, layer_0, layer_1, layer_2,
    #                                 os.path.join(args.results_dir, filename),
    #                                 save_it=not args.show, plot_GAN=args.plot_GAN,
    #                                 flow_thres=0., geant_thres=0.)
    #
    #for layer_id, layer_content in enumerate([layer_0, layer_1, layer_2]):
    #    filename = 'lateral_width_layer_'+str(layer_id) +('_GAN' if args.plot_GAN else '')+'.pdf'
    #    plot_layer_lateral_width(particles, flow_type, layer_content,
    #                             os.path.join(args.results_dir, filename), layer_id,
    #                             save_it=not args.show, plot_GAN=args.plot_GAN,
    #                             flow_thres=0., geant_thres=0.)
    #
    #filename = 'depth_weighted_energy_normed'+('_GAN' if args.plot_GAN else '')+'.pdf'
    #plot_depth_weighted_energy_normed(particles, flow_type, layer_0, layer_1, layer_2,
    #                                  os.path.join(args.results_dir, filename),
    #                                  save_it=not args.show, plot_GAN=args.plot_GAN,
    #                                  flow_thres=0., geant_thres=0.)
    #
    #filename = 'depth_weighted_energy_normed_std'+('_GAN' if args.plot_GAN else '')+'.pdf'
    #plot_depth_weighted_energy_normed_std(particles, flow_type, layer_0, layer_1, layer_2,
    #                                      os.path.join(args.results_dir, filename),
    #                                      save_it=not args.show, plot_GAN=args.plot_GAN,
    #                                      flow_thres=0., geant_thres=0.)

    #
    #for layer_id, layer_content in enumerate([layer_0, layer_1, layer_2]):
    #    for which_voxel in [1, 2, 3, 4, 5]:
    #        filename = str(which_voxel)+'_brightest_voxel_layer_'+str(layer_id)+\
    #            ('_GAN' if args.plot_GAN else '')+'.png'
    #        plot_brightest_voxel(particles, flow_type, layer_content,
    #                             os.path.join(args.results_dir, filename), layer_id,
    #                             which_voxel=which_voxel, save_it=not args.show,
    #                             plot_GAN=args.plot_GAN, flow_thres=0., geant_thres=0.)
    #
    #for scan_dir in ['eta', 'phi']:
    #    filename = 'centroid_corr_0_1_'+scan_dir+('_GAN' if args.plot_GAN else '')+'.png'
    #    plot_centroid_correlation(particles, flow_type, 0, layer_0, 1, layer_1,
    #                              os.path.join(args.results_dir, filename),
    #                              scan=scan_dir,
    #                              save_it=not args.show,
    #                              plot_GAN=args.plot_GAN,
    #                              flow_thres=0., geant_thres=0.)
    #    filename = 'centroid_corr_0_2_'+scan_dir+('_GAN' if args.plot_GAN else '')+'.png'
    #    plot_centroid_correlation(particles, flow_type, 0, layer_0, 2, layer_2,
    #                              os.path.join(args.results_dir, filename),
    #                              scan=scan_dir,
    #                              save_it=not args.show,
    #                              plot_GAN=args.plot_GAN,
    #                              flow_thres=0., geant_thres=0.)
    #    filename = 'centroid_corr_1_2_'+scan_dir+('_GAN' if args.plot_GAN else '')+'.png'
    #    plot_centroid_correlation(particles, flow_type, 1, layer_1, 2, layer_2,
    #                              os.path.join(args.results_dir, filename),
    #                              scan=scan_dir,
    #                              save_it=not args.show,
    #                              plot_GAN=args.plot_GAN,
    #                              flow_thres=0., geant_thres=0.)
    #for scan_dir in ['eta', 'phi']:
    #    for layer_id, layer_content in enumerate([layer_0, layer_1, layer_2]):
    #        filename = 'centroid_'+str(layer_id)+'_'+scan_dir+('_GAN' if args.plot_GAN else '')+\
    #            '.png'
    #        plot_centroid(particles, flow_type, layer_id, layer_content,
    #                      os.path.join(args.results_dir, filename),
    #                      scan=scan_dir, save_it=not args.show, plot_GAN=args.plot_GAN,
    #                      flow_thres=0., geant_thres=0.)


    # plot legend only (ignore popping up histo):
    if args.no_legend:
        handles, labels, entries = plot_depth_weighted_energy_normed_std(particles, flow_type, layer_0,
                                                                         layer_1, layer_2,
                                                                         'dummy', save_it=False,
                                                                         plot_GAN=args.plot_GAN,
                                                                         flow_thres=0.,
                                                                         geant_thres=0.)
        filename = 'legend'+('_GAN' if args.plot_GAN else '')+'.pdf'
        plot_legend(handles, labels, entries, os.path.join(args.results_dir, filename),
                    save_it=not args.show)


    # plotting more plots in single png file:
    matplotlib.rcParams.update({'font.size': 28})
    layer_list = [layer_0, layer_1, layer_2]

    ## plot selection of histos sensitive to flow I (old selection, not used)

    #plt.figure(figsize=(30, 30), dpi=300)
    #for i in range(3):
    #    plt.subplot(3, 3, i+1)
    #    plot_layer_energy(particles, flow_type, layer_list[i], 'dummy',
    #                      i, save_it=False, plot_GAN=args.plot_GAN,
    #                      flow_thres=0., geant_thres=0., is_sub=True)
    #for i in range(3):
    #    plt.subplot(3, 3, i+4)
    #    plot_energy_fraction(particles, flow_type, layer_0, layer_1, layer_2, 'dummy',
    #                         i, save_it=False, plot_GAN=args.plot_GAN, use_log=True,
    #                         flow_thres=0., geant_thres=0., is_sub=True)
    #plt.subplot(3, 3, 7)
    #plot_total_energy(particles, flow_type, layer_0, layer_1, layer_2, 'dummy',
    #                  save_it=False, plot_GAN=args.plot_GAN, flow_thres=0., geant_thres=0.,
    #                  is_sub=True)
    #plt.subplot(3, 3, 8)
    #plot_depth_weighted_total_energy(particles, flow_type, layer_0, layer_1, layer_2, 'dummy',
    #                                 save_it=False, plot_GAN=args.plot_GAN,
    #                                 flow_thres=0., geant_thres=0., is_sub=True)
    #plt.subplot(3, 3, 9)
    #plot_depth_weighted_energy_normed(particles, flow_type, layer_0, layer_1, layer_2, 'dummy',
    #                                  save_it=False, plot_GAN=args.plot_GAN,
    #                                  flow_thres=0., geant_thres=0., is_sub=True)
    #plt.subplots_adjust(wspace=0.25, hspace=0.3)
    #filename = 'flow1_histos'+('_GAN' if args.plot_GAN else '')+'.png'
    #plt.savefig(os.path.join(args.results_dir, filename))

    # plot one selection of histos sensitive to flow II
    #plt.figure(figsize=(30, 30), dpi=300)
    #for i in range(3):
    #    plt.subplot(3, 3, i+1)
    #    plot_layer_E_ratio(particles, flow_type, layer_list[i], 'dummy',
    #                       i, save_it=False, plot_GAN=args.plot_GAN,
    #                       flow_thres=0., geant_thres=0., is_sub=True)
    #    plt.subplot(3, 3, i+4)
    #    plot_layer_lateral_width(particles, flow_type, layer_list[i], 'dummy', i,
    #                             save_it=False, plot_GAN=args.plot_GAN,
    #                             flow_thres=0., geant_thres=0., is_sub=True)
    #    plt.subplot(3, 3, i+7)
    #    plot_layer_sparsity(particles, flow_type, layer_list[i], 'dummy',
    #                        i, save_it=False, plot_GAN=args.plot_GAN,
    #                        flow_thres=0., geant_thres=0., is_sub=True)
    #plt.subplots_adjust(wspace=0.25, hspace=0.3)
    #filename = 'flow2_histos'+('_GAN' if args.plot_GAN else '')+'.png'
    #plt.savefig(os.path.join(args.results_dir, filename))

    # plot brightest 3 voxel
    #plt.figure(figsize=(30, 30), dpi=300)
    #for i in range(3):
    #    for j in range(3):
    #        plt.subplot(3, 3, (3*i) + j + 1)
    #        plot_brightest_voxel(particles, flow_type, layer_list[j], 'dummy', j,
    #                             which_voxel=i+1, save_it=False,
    #                             plot_GAN=args.plot_GAN, flow_thres=0., geant_thres=0.,
    #                             is_sub=True)
    #plt.subplots_adjust(wspace=0.25, hspace=0.3)
    #filename = 'brightest_histos'+('_GAN' if args.plot_GAN else '')+'.png'
    #plt.savefig(os.path.join(args.results_dir, filename))

    # plot only centroid differences
    #plt.figure(figsize=(30, 20), dpi=300)
    #for index, scan_dir in enumerate(['eta', 'phi']):
    #    plt.subplot(2, 3, index*3 + 1)
    #    plot_centroid_correlation(particles, flow_type, 0, layer_0, 1, layer_1, 'dummy',
    #                              scan=scan_dir, save_it=False, plot_GAN=args.plot_GAN,
    #                              flow_thres=0., geant_thres=0., is_sub=True)
    #    plt.subplot(2, 3, index*3 + 2)
    #    plot_centroid_correlation(particles, flow_type, 0, layer_0, 2, layer_2, 'dummy',
    #                              scan=scan_dir, save_it=False, plot_GAN=args.plot_GAN,
    #                              flow_thres=0., geant_thres=0., is_sub=True)
    #    plt.subplot(2, 3, index*3 + 3)
    #    plot_centroid_correlation(particles, flow_type, 1, layer_1, 2, layer_2, 'dummy',
    #                              scan=scan_dir, save_it=False, plot_GAN=args.plot_GAN,
    #                              flow_thres=0., geant_thres=0., is_sub=True)
    #plt.subplots_adjust(wspace=0.25, hspace=0.3)
    #filename = 'centroid_corr'+('_GAN' if args.plot_GAN else '')+'.png'
    #plt.savefig(os.path.join(args.results_dir, filename))

    ## plot 3 brightest voxel and then centroids
    #plt.figure(figsize=(30, 40), dpi=300)
    #for i in range(2):
    #    for j in range(3):
    #        plt.subplot(4, 3, (3*i) + j + 1)
    #        plot_brightest_voxel(particles, flow_type, layer_list[j], 'dummy', j,
    #                             which_voxel=i+1, save_it=False,
    #                             plot_GAN=args.plot_GAN, flow_thres=0., geant_thres=0.,
    #                             is_sub=True)
    #
    #for index, scan_dir in enumerate(['eta', 'phi']):
    #    for layer_id, layer_content in enumerate([layer_0, layer_1, layer_2]):
    #        plt.subplot(4, 3, 7+index*3+layer_id)
    #        plot_centroid(particles, flow_type, layer_id, layer_content, 'dummy',
    #                      scan=scan_dir, save_it=False, plot_GAN=args.plot_GAN,
    #                      flow_thres=0., geant_thres=0., is_sub=True)
    #
    #plt.subplots_adjust(wspace=0.25, hspace=0.3)
    #filename = 'brightest_and_centroid_corr'+('_GAN' if args.plot_GAN else '')+'.png'
    #plt.savefig(os.path.join(args.results_dir, filename))


    ###   refined order of plots (used for the paper)   ###

    # plot selection of histos sensitive to flow I
    plt.figure(figsize=(40, 10), dpi=300)
    for i in range(3):
        plt.subplot(1, 4, i+1)
        plot_layer_energy(particles, flow_type, layer_list[i], 'dummy',
                          i, save_it=False, plot_GAN=args.plot_GAN,
                          flow_thres=0., geant_thres=0., is_sub=True)
    plt.subplot(1, 4, 4)
    plot_total_energy(particles, flow_type, layer_0, layer_1, layer_2, 'dummy',
                      save_it=False, plot_GAN=args.plot_GAN, flow_thres=0., geant_thres=0.,
                      is_sub=True)
    plt.subplots_adjust(left=0.05, right=0.95, wspace=0.25, hspace=0.3)
    filename = 'flow1_energy_histos'+('_GAN' if args.plot_GAN else '')+'.pdf'
    plt.savefig(os.path.join(args.results_dir, filename))

    plt.figure(figsize=(30, 20), dpi=300)
    for i in range(3):
        plt.subplot(2, 3, i+1)
        plot_energy_fraction(particles, flow_type, layer_0, layer_1, layer_2, 'dummy',
                             i, save_it=False, plot_GAN=args.plot_GAN, use_log=True,
                             flow_thres=0., geant_thres=0., is_sub=True)
    plt.subplot(2, 3, 4)
    plot_depth_weighted_total_energy(particles, flow_type, layer_0, layer_1, layer_2, 'dummy',
                                     save_it=False, plot_GAN=args.plot_GAN,
                                     flow_thres=0., geant_thres=0., is_sub=True)
    plt.subplot(2, 3, 5)
    plot_depth_weighted_energy_normed(particles, flow_type, layer_0, layer_1, layer_2, 'dummy',
                                      save_it=False, plot_GAN=args.plot_GAN,
                                      flow_thres=0., geant_thres=0., is_sub=True)
    plt.subplot(2, 3, 6)
    plot_depth_weighted_energy_normed_std(particles, flow_type, layer_0, layer_1, layer_2, 'dummy',
                                          save_it=False, plot_GAN=args.plot_GAN,
                                          flow_thres=0., geant_thres=0., is_sub=True)
    filename = 'flow1_shower_histos'+('_GAN' if args.plot_GAN else '')+'.pdf'
    plt.subplots_adjust(wspace=0.25, hspace=0.3, left=0.1, right=0.9)
    plt.savefig(os.path.join(args.results_dir, filename))

    # plot selection of histos sensitive to flow II
    plt.figure(figsize=(30, 40), dpi=300)
    for i in range(3):
        plt.subplot(4, 3, i+1)
        plot_brightest_voxel(particles, flow_type, layer_list[i], 'dummy', i,
                             which_voxel=1, save_it=False,
                             plot_GAN=args.plot_GAN, flow_thres=0., geant_thres=0.,
                             is_sub=True)
        plt.subplot(4, 3, i+4)
        plot_brightest_voxel(particles, flow_type, layer_list[i], 'dummy', i,
                             which_voxel=2, save_it=False,
                             plot_GAN=args.plot_GAN, flow_thres=0., geant_thres=0.,
                             is_sub=True)
        plt.subplot(4, 3, i+7)
        plot_layer_E_ratio(particles, flow_type, layer_list[i], 'dummy',
                           i, save_it=False, plot_GAN=args.plot_GAN,
                           flow_thres=0., geant_thres=0., is_sub=True)
        plt.subplot(4, 3, i+10)
        plot_layer_sparsity(particles, flow_type, layer_list[i], 'dummy',
                            i, save_it=False, plot_GAN=args.plot_GAN,
                            flow_thres=0., geant_thres=0., is_sub=True)
    plt.subplots_adjust(wspace=0.25, hspace=0.3, left=0.1, right=0.9)
    filename = 'flow2_voxel_histos'+('_GAN' if args.plot_GAN else '')+'.pdf'
    plt.savefig(os.path.join(args.results_dir, filename))

    plt.figure(figsize=(30, 30), dpi=300)
    for layer_id, layer_content in enumerate([layer_0, layer_1, layer_2]):
        plt.subplot(3, 3, layer_id+1)
        plot_centroid(particles, flow_type, layer_id, layer_content, 'dummy', scan='phi',
                      save_it=False, plot_GAN=args.plot_GAN, flow_thres=0., geant_thres=0.,
                      is_sub=True)
        plt.subplot(3, 3, layer_id+4)
        plot_centroid(particles, flow_type, layer_id, layer_content, 'dummy', scan='eta',
                      save_it=False, plot_GAN=args.plot_GAN, flow_thres=0., geant_thres=0.,
                      is_sub=True)
        plt.subplot(3, 3, layer_id+7)
        plot_layer_lateral_width(particles, flow_type, layer_content, 'dummy', layer_id,
                                 save_it=False, plot_GAN=args.plot_GAN,
                                 flow_thres=0., geant_thres=0., is_sub=True)
    plt.subplots_adjust(wspace=0.25, hspace=0.3, left=0.1, right=0.9)
    filename = 'flow2_centroid_histos'+('_GAN' if args.plot_GAN else '')+'.pdf'
    plt.savefig(os.path.join(args.results_dir, filename))
