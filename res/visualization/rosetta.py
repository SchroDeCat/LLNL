import numpy as np
from matplotlib import pyplot as plt

n_repeat = 10
n_iter = 100
n_init = 10
sqrt_n = np.sqrt(n_repeat)
batch_size=1

# []:,n_init:]
RES_num = {}
# RES_num["dkbo"] = np.load("../batch/Oct_x_Rosetta-none-ts-R10-P1-T10_I1_L2-TI10-USexact.npy")
RES_num["DKBO-AE"] = np.load("../batch/Oct_x_Rosetta-none-ts-R10-P1-T10_I1_L2-TI10-USexact-AE.npy")
# RES_num[f"DKBO-OLP"] = np.load(f"../batch/OL-batched_oct_x_Rosetta-B1-kmeans-y-ts-R10-P3-T200_I0_L2-TI10-USmax.npy")
# RES_num[f"DKBO-Filter"] = np.load(f"../filter/OL-filtering_oct_x_Rosetta-none-ts-R10-P2-T200_I10_L2-TI10-USexact.npy")
RES_num[f"BALLET"]= np.load(f"../filter/OL-filtering_oct_x_Rosetta-1.0-none-ts-R10-P2-T100_I10_L2-TI10-USexact.npy")
RES_num["LA-MCTS"] = np.load("../lamcts/lamcts-dkbo-oct_x_Rosetta-R10-P1-T100.npy")[:,n_init:]
RES_num["TuRBO-DK"] = np.load("../turbo/turbo-oct_x_Rosetta-R10-T100.npy")[:,n_init:]
    
n_iter = n_iter - n_init
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
plt.savefig("Rosetta.png")