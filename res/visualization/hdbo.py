import numpy as np
from matplotlib import pyplot as plt

n_repeat = 10
n_iter=30
n_init=10
sqrt_n = np.sqrt(n_repeat)

RES_num = {}

# RES_num["dkbo"] = np.load("OL-batched_nano-none-ts-R10-P1-T10_I1_L2-TI10-USexact.npy")
# RES_num["DKBO-ae"] = np.load("/Users/ttqsfzx/Documents/Research/LOCo/loco/LOCo/res/UAI2022-rebuttal/200d-ucb-i10-r5-h50-k50-lr0-rho1-f1-ae-fix.npy")
# RES_num["DKBO-AE"] = np.load("/Users/ttqsfzx/Documents/Research/LLNL/LLNL/res/batch/OL-normal_hdbo200-none-ts-R10-P1-T10_I1_L6-TI10-USexact.npy")
RES_num["DKBO-AE"] = np.load("../old-dataset/200d-ucb-i10-r5-h50-k50-lr0-rho1-f1-ae-fix.npy")
# RES_num[f"DKBO-OLP"] = np.load(f"../batch/OL-batched_hdbo200-B1-kmeans-y-ts-R10-P3-T30_I0_L2-TI10-USmax.npy")
# RES_num[f"DKBO-Filter"] = np.load(f"../filter/OL-filtering_hdbo200-none-ts-R10-P2-T30_I10_L2-TI10-USexact.npy")
RES_num[f"BALLET"] = np.load(f"../filter/OL-filtering_hdbo200-none-ts-R10-P2-T30_I10_L2-TI10-USexact.npy")
# RES_num[f"BALLET"] = np.load("../filter/study_partition/OL-filtering_hdbo200-ae-retrained-10.0-none-ucb-R10-P2-T20_I30_L2-TI10-USexact.npy")
# RES_num[f"BALLET"] = np.load("../filter/study_partition/OL-filtering_hdbo200-ae-retrained-10.0-none-ucb-R10-P2-T30_I25_L2-TI10-USexact.npy")
# RES_num[f"BALLET"] = np.load("../filter/study_partition/OL-filtering_hdbo200-regularized-5.0-none-ucb-R10-P2-T20_I5_L2-TI10-USexact.npy")
# RES_num[f"BALLET"] = np.load("/Users/ttqsfzx/Documents/Research/LLNL/LLNL/res/filter/debug/OL-filtering_hdbo200-sgld-highd-10.0-none-ucb-R10-P2-T20_I5_L4-TI1000-USexact.npy")
# RES_num[f"BALLET"] = np.load("/Users/ttqsfzx/Documents/Research/LLNL/LLNL/res/filter/debug/OL-filtering_hdbo200-sgld-highd-10.0-none-ucb-R10-P2-T20_I5_L6-TI10-USexact.npy")

# RES_num[f"BALLET"] = np.load("../filter/debug/OL-filtering_hdbo200-sgld-highd-2.0-none-ucb-R10-P2-T20_I5_L6-TI10-USexact.npy")
# RES_num[f"BALLET-LOCo"] = np.load("../filter/debug/OL-filtering_hdbo200-sgld-highd-loco-2.0-none-ucb-R10-P2-T20_I5_L6-TI100-USexact.npy")
# RES_num[f"BALLET-Spectrum"] = np.load("../filter/debug/OL-filtering_hdbo200-sgld-highd-spectrum-2.0-none-ucb-R10-P2-T20_I5_L6-TI10-USexact.npy")
# RES_num[f"BALLET"] = np.load(f"../filter/OL-filtering_hdbo200_debug-1.0-none-ts-R10-P2-T30_I10_L2-TI10-USexact.npy")
RES_num["LA-MCTS"] = np.load("../lamcts/lamcts-dkbo-hdbo200-R10-P1-T30.npy")[:,n_init:]
RES_num["TuRBO-DK"] = np.load("../turbo/turbo-hdbo200-R10-T30.npy")[:,n_init:]


n_iter -= n_init
init_regret = RES_num["DKBO-AE"][:,0].mean(axis=0)

for method in RES_num.keys():
# for method in ["dkbo",'DKBO-OLP-Batch-5']:
    # CI = 1.96
    CI = 1
    coef = CI /sqrt_n
    RES_num[method][:,0] = init_regret
    plt.plot(RES_num[method][:,:n_iter].mean(axis=0), label=method)
    plt.fill_between(np.arange(n_iter), RES_num[method][:,:n_iter].mean(axis=0) - RES_num[method][:,:n_iter].std(axis=0) * coef, 
                     RES_num[method][:,:n_iter].mean(axis=0) + RES_num[method][:,:n_iter].std(axis=0) * coef, alpha=0.3)

# plt.title("Simple Regret Comparison")
plt.legend()
plt.xlabel("Iteration", fontsize=20)
# plt.ylabel("Simple Regret")
# plt.show()
plt.savefig("HDBO-SLGD.png")