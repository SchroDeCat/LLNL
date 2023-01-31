import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

fontsize = 18
n_init = 10

fig = plt.figure(figsize=[18, 10])

def plot_subfigure(ax, RES_num, name):
    init_regret = RES_num["DKBO-AE"][:,0].mean(axis=0)
    n_iter_list = []
    for method in RES_num.keys():
        CI = 1
        n_repeat = RES_num[method].shape[0]
        n_iter = RES_num[method].shape[1]
        n_iter_list.append(n_iter)
        sqrt_n = np.sqrt(n_repeat)
        coef = CI /sqrt_n
        RES_num[method][:,0] = init_regret
        RES_num[method] = np.minimum.accumulate(RES_num[method], axis=1)
        if method in ["BALLET-ICI"]:
            _label = f"{method} (R+G)"
        elif method in ["DKBO-AE"]:
            _label = f"{method} (G)"
        elif method in ["BALLET-RTS", "BALLET-RCI"]:
            _label = f"{method} (R)"
        elif method in ["LA-MCTS", "TuRBO-DK"]:
            _label = f"{method} (R)"
        else:
            raise NotImplementedError(f"unknown method {method}")
        ax.plot(RES_num[method].mean(axis=0), label=_label, alpha=1.0 if method == "BALLET-ICI" else 0.6, lw=3 if method == "BALLET-ICI" else 2)
        ax.fill_between(np.arange(n_iter), RES_num[method].mean(axis=0) - RES_num[method].std(axis=0) * coef, 
                        RES_num[method].mean(axis=0) + RES_num[method].std(axis=0) * coef, alpha=0.3)
    # ax.set_xlabel("Iteration", fontsize=fontsize)
    ax.set_xlabel(name, fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.set_xlim([0, min(n_iter_list)-1])
    # ax.set_aspect('equal')

# hdbo
RES_num = {}
RES_num["DKBO-AE"] = np.load("./ci/Pure-DK-hdbob0-ci-R10-T100_L4-TI10.npy")
RES_num[f"BALLET-ICI"] =  np.load("./ci/OL-hdbo-B0-FB0.2-none-ci-R10-P2-T100_I10_L4-TI400-USexact-sec.npy")
RES_num[f"BALLET-RTS"] = np.load("./ts/OL-hdbo-B0-FB0.2-none-ts-R10-P2-T100_I10_L4-TI10-USexact.npy")
RES_num[f"BALLET-RCI"] = np.load("./ci/OL-hdbo-hd-B0-FB0.2-none-ci-R10-P2-T100_I10_L4-TI10-USexact.npy")
RES_num["LA-MCTS"] = np.load("./lamcts/lamcts-dkbo-hdbo200-R10-P1-T50.npy")[:,n_init:]
RES_num["TuRBO-DK"] = np.load("./turbo/turbo-hdbo200-R10-T50.npy")[:,n_init:]

ax = plt.subplot(2,3,2)

plot_subfigure(ax, RES_num, "(b) HDBO")

# eg1d
RES_num = {}
RES_num["DKBO-AE"] = np.load("./ci/Pure-DK-eg1db0-ci-R30-T50_L4-TI10.npy")
RES_num[f"BALLET-ICI"] =  np.load("./ci/OL-eg1d-B0-FB0.2-none-ci-R10-P2-T50_I5_L4-TI10-USexact-sec.npy")
RES_num[f"BALLET-RTS"] = np.load("./ts/OL-eg1d-B0-FB0.2-none-ts-R10-P2-T50_I5_L4-TI10-USexact.npy")
RES_num[f"BALLET-RCI"] = np.load("./ci/OL-eg1d-B0-FB0.2-none-CI-R10-P2-T50_I5_L4-TI10-USexact.npy")
RES_num["LA-MCTS"] = np.load("./lamcts/lamcts-dkbo-OneDeg-R10-P1-T50.npy")[:,n_init:]
RES_num["TuRBO-DK"] = np.load("./turbo/turbo-OneDeg-R10-T50.npy")[:,n_init:]

ax = plt.subplot(2,3,1)

plot_subfigure(ax, RES_num, "(a) 1D Toy")

# nano
RES_num = {}
RES_num["DKBO-AE"] = np.load("./ci/Pure-DK-nanob0_hd-ci-R30-T100_L4-TI10.npy")
RES_num[f"BALLET-ICI"] = np.load(f"./ci/OL-nano-hd-B0-FB0.8-none-ci-R30-P2-T100_I10_L4-TI100-USexact-sec.npy")
RES_num[f"BALLET-RTS"] = np.load(f"./ts/OL-nano-hd-B0-FB0.8-none-ts-R30-P2-T100_I10_L4-TI100-USexact.npy")
RES_num[f"BALLET-RCI"] = np.load(f"./ci/OL-nano-hd-B0-FB0.8-none-ci-R30-P2-T100_I10_L4-TI100-USexact.npy")
RES_num["LA-MCTS"] = np.load("./lamcts/lamcts-dkbo-hd-nano-R10-P1-T100.npy")[:,n_init:]
RES_num["TuRBO-DK"] = np.load("./turbo/turbo-hd-nano-R10-T100.npy")[:,n_init:]

ax = plt.subplot(2,3,3)

plot_subfigure(ax, RES_num, "(c) Nanophotonic")


# #rosetta
RES_num = {}
RES_num["DKBO-AE"] = np.load("./ci/Pure-DK-rosettab0-ci-R30-T100_L4-TI10.npy")
RES_num[f"BALLET-ICI"] = np.load(f"./ci/OL-rosetta-B0-FB0.2-none-ci-R10-P2-T100_I10_L4-TI10-USexact-sec.npy")
RES_num[f"BALLET-RTS"] = np.load(f"./ts/OL-rosetta-B0-FB0.2-none-ts-R10-P2-T100_I10_L4-TI10-USexact.npy")
RES_num[f"BALLET-RCI"] = np.load(f"./ci/OL-rosetta-B0-FB0.2-none-ci-R10-P2-T100_I10_L4-TI10-USexact.npy")
RES_num["LA-MCTS"] = np.load("./lamcts/lamcts-dkbo-oct_x_Rosetta-R10-P1-T100.npy")[:,n_init:]
RES_num["TuRBO-DK"] = np.load("./turbo/turbo-bo-oct_x_Rosetta-R10-T100.npy")[:,n_init:]

ax = plt.subplot(2,3,6)

plot_subfigure(ax, RES_num, "(f) Rosetta")


# water converter
RES_num = {}
RES_num["DKBO-AE"] = np.load("./ci/Pure-DK-water_converterb0_hd-ci-R30-T100_L4-TI10.npy")
RES_num[f"BALLET-ICI"] = np.load(f"./ci/OL-water_converter-hd-B0-FB80.0-none-ci-R10-P2-T100_I10_L4-TI100-USexact-sec.npy")
RES_num[f"BALLET-RTS"] = np.load(f"./ts/OL-water_converter-hd-B0-FB80.0-none-ts-R10-P2-T100_I10_L4-TI100-USexact.npy")
RES_num[f"BALLET-RCI"] = np.load(f"./ci/OL-water_converter-hd-B0-FB80-none-ci-R10-P2-T100_I10_L4-TI100-USexact.npy")
RES_num["LA-MCTS"] = np.load("./lamcts/lamcts-dkbo-hd-water_converter-R10-P1-T100.npy")[:,n_init:]
RES_num["TuRBO-DK"] = np.load("./turbo/turbo-hd-water_converter-R10-T100.npy")[:,n_init:]

ax = plt.subplot(2,3,4)

plot_subfigure(ax, RES_num, "(d) Water Converter")

# GB1
RES_num = {}
RES_num["DKBO-AE"] = np.load("./ci/Pure-DK-gb1b0-ci-R30-T100_L4-TI10.npy")
RES_num[f"BALLET-ICI"] = np.load(f"./ci/OL-gb1-B0-FB0.2-none-ci-R10-P2-T100_I10_L4-TI10-USexact-sec.npy")
RES_num[f"BALLET-RTS"] = np.load(f"./ts/OL-gb1-B0-FB0.2-none-ts-R10-P2-T100_I10_L4-TI10-USexact.npy")
RES_num[f"BALLET-RCI"] = np.load(f"./ci/OL-gb1-B0-FB0.2-none-ci-R10-P2-T100_I10_L4-TI10-USexact.npy")
RES_num["LA-MCTS"] = np.load("./lamcts/lamcts-dkbo-gb1_embed-R10-P1-T100.npy")[:,n_init:]
RES_num["TuRBO-DK"] = np.load("./turbo/turbo-gb1_embed-R10-T100.npy")[:,n_init:]

ax = plt.subplot(2,3,5)

plot_subfigure(ax, RES_num, "(e) GB1")

handles, labels = ax.get_legend_handles_labels()
plt.tight_layout()
fig.legend(handles, labels, loc='upper center',bbox_to_anchor=(0.5, 1.05), ncol=len(labels), prop={'size': fontsize})
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
plt.savefig("icml_1.pdf", bbox_inches='tight')
plt.savefig("icml_1.png", bbox_inches='tight')
plt.close()
# plt.show()