import numpy as np
from matplotlib import pyplot as plt

n_repeat = 10
n_iter=200
sqrt_n = np.sqrt(n_repeat)

RES_num = {}
# RES_num["dkbo"] = np.load("OL-batched_nano-none-ts-R10-P1-T10_I1_L2-TI10-USexact.npy")
# RES_num["dkbo"] = np.load("OL-batched_nano-none-ts-R30-P1-T10_I1_L2-TI10-USexact.npy") # r30
# RES_num["dkbo-ae"] = np.load("OL-batched_nano-none-ts-R10-P1-T10_I1_L2-TI10-USexact-AE.npy")
# RES_num["dkbo-ae3"] = np.load("OL-batched_nano-none-ts-R30-P1-T10_I1_L2-TI10-USexact-AE.npy") # r30


for batch_size in [1,2,4]:
    RES_num[f"DKBO-OLP-Batch-{batch_size}"] = np.load(f"OL-batched_water_converter-B{batch_size}-kmeans-y-ts-R10-P3-T200_I0_L2-TI10-USmax.npy")
    


for method in RES_num.keys():
# for method in ["dkbo",'DKBO-OLP-Batch-5']:
    CI = 1.96
    coef = CI /sqrt_n
    plt.plot(RES_num[method].mean(axis=0), label=method)
    print(f"{method} {RES_num[method][:100].mean()}")
    # plt.fill_between(np.arange(n_iter), RES_num[method].mean(axis=0) - RES_num[method].std(axis=0) * coef, 
                    #  RES_num[method].mean(axis=0) + RES_num[method].std(axis=0) * coef, alpha=0.3)

plt.title("Simple Regret Comparison")
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("Simple Regret")
# plt.show()
plt.savefig("SimpleRegret_water_converter.png")