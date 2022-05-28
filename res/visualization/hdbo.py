import numpy as np
from matplotlib import pyplot as plt

n_repeat = 10
n_iter=30
sqrt_n = np.sqrt(n_repeat)

RES_num = {}

# RES_num["dkbo"] = np.load("OL-batched_nano-none-ts-R10-P1-T10_I1_L2-TI10-USexact.npy")
RES_num["DKBO-ae"] = np.load("/Users/ttqsfzx/Documents/Research/LOCo/loco/LOCo/res/UAI2022-rebuttal/200d-ucb-i10-r5-h50-k50-lr0-rho1-f1-ae-fix.npy")
RES_num[f"DKBO-OLP"] = np.load(f"../batch/OL-batched_hdbo200-B1-kmeans-y-ts-R10-P3-T30_I0_L2-TI10-USmax.npy")
RES_num[f"DKBO-Filter"] = np.load(f"../filter/OL-filtering_hdbo200-none-ts-R10-P2-T30_I10_L2-TI10-USexact.npy")
RES_num["LA-MCTS"] = np.load("../lamcts/lamcts-dkbo-hdbo200-R10-P1-T30.npy")
RES_num["TurBO-DK"] = np.load("../turbo/turbo-hdbo200-R10-T30.npy")

for method in RES_num.keys():
    print(f"method {method}")
    RES_num[method]
    # CI = 1.96
    CI = 1
    coef = CI /sqrt_n
    plt.plot(RES_num[method][:,:n_iter].mean(axis=0), label=method)
    plt.fill_between(np.arange(n_iter), RES_num[method][:,:n_iter].mean(axis=0) - RES_num[method][:,:n_iter].std(axis=0) * coef, 
                     RES_num[method][:,:n_iter].mean(axis=0) + RES_num[method][:,:n_iter].std(axis=0) * coef, alpha=0.3)

plt.title("Simple Regret Comparison")
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("Simple Regret")
plt.savefig("HDBO.png")