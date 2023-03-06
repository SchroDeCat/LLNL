import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

fontsize = 18
n_init = 10

fig = plt.figure(figsize=[18, 6])

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
        elif method in ["DKBO-AE", "Exact-GP"]:
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

# eg1d
RES_num = {}
RES_num["Exact-GP"] = np.load("./ts/Pure-DK-eg1db0-Exact-ts-R9-T50_L4-TI10.npy")
RES_num["DKBO-AE"] = np.load("./ci/Pure-DK-eg1db0-ci-R30-T50_L4-TI10.npy")
RES_num[f"BALLET-ICI"] =  np.load("./ci/OL-eg1d-B0-FB0.2-none-ci-R10-P2-T50_I5_L4-TI10-USexact-sec.npy")
RES_num[f"BALLET-RTS"] = np.load("./ts/OL-eg1d-B0-FB0.2-none-ts-R10-P2-T50_I5_L4-TI10-USexact.npy")
RES_num[f"BALLET-RCI"] = np.load("./ci/OL-eg1d-B0-FB0.2-none-CI-R10-P2-T50_I5_L4-TI10-USexact.npy")

ax = plt.subplot(1,3,1)

plot_subfigure(ax, RES_num, "(a) 1D Toy")

# nano
RES_num = {}
RES_num["Exact-GP"] = np.load("../icml/ts/Pure-DK-nanob0_hd-Exact-ts-R4-T100_L4-TI10.npy")[:,:-10]
RES_num["DKBO-AE"] = np.load("./ci/Pure-DK-nanob0_hd-ci-R30-T100_L4-TI10.npy")[:,:-10]
RES_num[f"BALLET-ICI"] = np.load(f"./ci/OL-nano-hd-B0-FB0.8-none-ci-R30-P2-T100_I10_L4-TI100-USexact-sec.npy")[:,:-10]
RES_num[f"BALLET-RTS"] = np.load(f"./ts/OL-nano-hd-B0-FB0.8-none-ts-R30-P2-T100_I10_L4-TI100-USexact.npy")[:,:-10]
RES_num[f"BALLET-RCI"] = np.load(f"./ci/OL-nano-hd-B0-FB0.8-none-ci-R30-P2-T100_I10_L4-TI100-USexact.npy")[:,:-10]

ax = plt.subplot(1,3,2)

plot_subfigure(ax, RES_num, "(c) Nanophotonic")


# water converter
RES_num = {}
RES_num["Exact-GP"] = np.load("../icml/ts/Pure-DK-water_converterb0_hd-Exact-ts-R4-T100_L4-TI10.npy")
RES_num["DKBO-AE"] = np.load("./ci/Pure-DK-water_converterb0_hd-ci-R30-T100_L4-TI10.npy")
RES_num[f"BALLET-ICI"] = np.load(f"./ci/OL-water_converter-hd-B0-FB80.0-none-ci-R10-P2-T100_I10_L4-TI100-USexact-sec.npy")
RES_num[f"BALLET-RTS"] = np.load(f"./ts/OL-water_converter-hd-B0-FB80.0-none-ts-R10-P2-T100_I10_L4-TI100-USexact.npy")
RES_num[f"BALLET-RCI"] = np.load(f"./ci/OL-water_converter-hd-B0-FB80-none-ci-R10-P2-T100_I10_L4-TI100-USexact.npy")

ax = plt.subplot(1,3,3)

plot_subfigure(ax, RES_num, "(d) Water Converter")

handles, labels = ax.get_legend_handles_labels()
plt.tight_layout()
fig.legend(handles, labels, loc='upper center',bbox_to_anchor=(0.5, 1.05), ncol=len(labels), prop={'size': fontsize})
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
plt.savefig("icml_exact_1.pdf", bbox_inches='tight')
plt.savefig("icml_exact_1.png", bbox_inches='tight')
plt.close()
# plt.show()