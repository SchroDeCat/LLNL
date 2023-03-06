import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

fontsize = 18
n_init = 10

fig = plt.figure(figsize=[18, 10])

def plot_subfigure(ax, RES_num, name, data_size=1000,_delta=0.2):
    n_iter_list = []
    for method in RES_num.keys():
        RES_num[method] = RES_num[method][:,:,n_init:]
        CI = 1
        n_repeat = RES_num[method].shape[0]
        n_iter = RES_num[method].shape[-1]
        sqrt_n = np.sqrt(n_repeat)
        coef = CI /sqrt_n
        # filter_interval=10 if n_iter >=100 else n_iter // 10
        filter_interval=1
        beta_correction = [2 / (2 * np.log((data_size * (np.pi * (iter + 1)) ** 2) /(6 * _delta))) ** 0.5 for iter in range(n_iter)][::filter_interval]
        print(n_iter, RES_num[method].shape, max(beta_correction))
        global_ci       = RES_num[method][:,0,::filter_interval]
        roi_ci          = RES_num[method][:,1,::filter_interval]
        intersect_ci    = RES_num[method][:,2,::filter_interval]
        bias            = np.min([roi_ci.mean(axis=0) - intersect_ci.mean(axis=0), np.zeros(intersect_ci.shape[-1])])
        intersect_ci    += bias
        

        for label, res in {"Global": global_ci, "ROI": roi_ci, "Intersection": intersect_ci}.items():
            res = res * np.array(beta_correction)
            n_iter_list.append(len(beta_correction))
            res_mean, res_std = res.mean(axis=0), res.std(axis=0)
            ax.plot(res_mean, label=label)
            ax.fill_between(np.arange(len(beta_correction)), res_mean - res_std * coef, 
                            res_mean + res_std * coef, alpha=0.3)
    # ax.set_xlabel("Iteration", fontsize=fontsize)
    ax.set_xlabel(name, fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.set_xlim([0, min(n_iter_list)-1])
    # ax.set_aspect('equal')

# hdbo
RES_num = {}
# RES_num["DKBO-AE"] = np.load("./ci/Pure-DK-hdbob0-ci-R10-T100_L4-TI10.npy")
# RES_num[f"BALLET"] =  np.load("./ci/OL-hdbo-B0-FB0.2-interval-none-ci-R10-P2-T100_I10_L4-TI400-USexact-sec.npy")
RES_num[f"BALLET"] = np.load("./ci/OL-hdbo-B0-FB0.2-interval-none-ci-R10-P2-T100_I1_L4-TI10-USexact-sec.npy")
# print(RES_num[f"BALLET-non-Intersection"][:,-1].mean())
# RES_num["LA-MCTS"] = np.load("./lamcts/lamcts-dkbo-hdbo200-R10-P1-T50.npy")[:,n_init:]
# RES_num["TuRBO-DK"] = np.load("./turbo/turbo-hdbo200-R10-T50.npy")[:,n_init:]

ax = plt.subplot(2,3,2)

plot_subfigure(ax, RES_num, "(b) HDBO", np.load("../../data/HDBO200.npy").shape[0])

# eg1d
RES_num = {}
# RES_num["DKBO-AE"] = np.load("./ci/Pure-DK-eg1db0-ci-R30-T50_L4-TI10.npy")
# RES_num[f"BALLET"] =  np.load("./ci/OL-eg1d-B0-FB0.2-none-ci-R10-P2-T50_I5_L4-TI10-USexact-sec.npy")
RES_num[f"BALLET-non-Intersection"] = np.load("./ci/OL-eg1d-B0-FB0.2-interval-none-CI-R10-P2-T50_I1_L4-TI10-USexact-sec.npy")
# RES_num["LA-MCTS"] = np.load("./lamcts/lamcts-dkbo-OneDeg-R10-P1-T50.npy")[:,n_init:]
# RES_num["TuRBO-DK"] = np.load("./turbo/turbo-OneDeg-R10-T50.npy")[:,n_init:]

ax = plt.subplot(2,3,1)

plot_subfigure(ax, RES_num, "(a) 1D Toy", np.load("../../data/opt_eg1d.npy").shape[0])

# nano
RES_num = {}
# RES_num["DKBO-AE"] = np.load("./ci/Pure-DK-nanob0_hd-ci-R30-T100_L4-TI10.npy")
# RES_num[f"BALLET"] = np.load(f"./ci/OL-nano-hd-B0-FB0.8-none-ci-R30-P2-T100_I10_L4-TI100-USexact-sec.npy")
RES_num[f"BALLET-non-Intersection"] = np.load(f"./ci/OL-nano-B0-FB0.2-interval-none-ci-R10-P2-T100_I1_L4-TI10-USexact-sec.npy")
# RES_num["LA-MCTS"] = np.load("./lamcts/lamcts-dkbo-hd-nano-R10-P1-T100.npy")[:,n_init:]
# RES_num["TuRBO-DK"] = np.load("./turbo/turbo-hd-nano-R10-T100.npy")[:,n_init:]

ax = plt.subplot(2,3,3)

plot_subfigure(ax, RES_num, "(c) Nanophotonic",  torch.load("../../data/data_nano_mf.pt").size(0))


# #rosetta
RES_num = {}
# RES_num["DKBO-AE"] = np.load("./ci/Pure-DK-rosettab0-ci-R30-T100_L4-TI10.npy")
# RES_num[f"BALLET"] = np.load(f"./ci/OL-rosetta-B0-FB0.2-none-ci-R10-P2-T100_I10_L4-TI10-USexact-sec.npy")
RES_num[f"BALLET-non-Intersection"] = np.load(f"./ci/OL-rosetta-B0-FB0.2-interval-none-ci-R10-P2-T100_I1_L4-TI10-USexact-sec.npy")
# RES_num["LA-MCTS"] = np.load("./lamcts/lamcts-dkbo-oct_x_Rosetta-R10-P1-T100.npy")[:,n_init:]
# RES_num["TuRBO-DK"] = np.load("./turbo/turbo-bo-oct_x_Rosetta-R10-T100.npy")[:,n_init:]

ax = plt.subplot(2,3,6)

plot_subfigure(ax, RES_num, "(f) Rosetta", torch.load("../../data/data_oct_x_to_Rosetta.pt").size(0))

ax.set_ylim([0, 4])


# water converter
RES_num = {}
# RES_num["DKBO-AE"] = np.load("./ci/Pure-DK-water_converterb0_hd-ci-R30-T100_L4-TI10.npy")
# RES_num[f"BALLET"] = np.load(f"./ci/OL-water_converter-hd-B0-FB80.0-none-ci-R10-P2-T100_I10_L4-TI100-USexact-sec.npy")
RES_num[f"BALLET-non-Intersection"] = np.load(f"./ci/OL-water_converter-B0-FB0.2-interval-none-CI-R10-P2-T100_I1_L4-TI10-USexact-sec.npy")
# RES_num["LA-MCTS"] = np.load("./lamcts/lamcts-dkbo-hd-water_converter-R10-P1-T100.npy")[:,n_init:]
# RES_num["TuRBO-DK"] = np.load("./turbo/turbo-hd-water_converter-R10-T100.npy")[:,n_init:]

ax = plt.subplot(2,3,4)

plot_subfigure(ax, RES_num, "(d) Water Converter", np.load("../../data/water_converter.npy").shape[0])

# GB1
RES_num = {}
# RES_num["DKBO-AE"] = np.load("./ci/Pure-DK-gb1b0-ci-R30-T100_L4-TI10.npy")
# RES_num[f"BALLET"] = np.load(f"./ci/OL-gb1-B0-FB0.2-none-ci-R10-P2-T100_I10_L4-TI10-USexact-sec.npy")
RES_num[f"BALLET-non-Intersection"] = np.load(f"./ci/OL-gb1-B0-FB0.2-interval-none-ci-R10-P2-T100_I1_L4-TI10-USexact-sec.npy")
# RES_num["LA-MCTS"] = np.load("./lamcts/lamcts-dkbo-gb1_embed-R10-P1-T100.npy")[:,n_init:]
# RES_num["TuRBO-DK"] = np.load("./turbo/turbo-gb1_embed-R10-T100.npy")[:,n_init:]

ax = plt.subplot(2,3,5)

plot_subfigure(ax, RES_num, "(e) GB1", np.load("../../data/gb1_embed.npy").shape[0])

handles, labels = ax.get_legend_handles_labels()
plt.tight_layout()
fig.legend(handles, labels, loc='upper center',bbox_to_anchor=(0.5, 1.05), ncol=len(labels), prop={'size': fontsize})
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
plt.savefig("aistats-interval.png", bbox_inches='tight')
plt.savefig("aistats-interval.pdf", bbox_inches='tight')
plt.close()
# plt.show()