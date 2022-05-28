import numpy as np
from matplotlib import pyplot as plt

n_repeat = 10
n_iter=100
sqrt_n = np.sqrt(n_repeat)
batch_size=1

RES_num = {}
RES_num["dkbo"] = np.load("../batch/Oct_x_Rosetta-none-ts-R10-P1-T10_I1_L2-TI10-USexact.npy")
RES_num["dkbo-ae"] = np.load("../batch/Oct_x_Rosetta-none-ts-R10-P1-T10_I1_L2-TI10-USexact-AE.npy")
RES_num[f"DKBO-OLP"] = np.load(f"../batch/OL-batched_oct_x_Rosetta-B1-kmeans-y-ts-R10-P3-T200_I0_L2-TI10-USmax.npy")
RES_num[f"DKBO-Filter"] = np.load(f"../filter/OL-filtering_oct_x_Rosetta-none-ts-R10-P2-T200_I10_L2-TI100-USexact.npy")
RES_num["LA-MCTS"] = np.load("../lamcts/lamcts-dkbo-oct_x_Rosetta-R10-P1-T100.npy")
RES_num["TurBO-DK"] = np.load("../turbo/turbo-oct_x_Rosetta-R10-T100.npy")
    

for method in RES_num.keys():
# for method in ["dkbo",'DKBO-OLP-Batch-5']:
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
# plt.show()
plt.savefig("Rosetta.png")