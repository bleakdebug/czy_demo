import matplotlib.pyplot as plt
import torch
import numpy as np
pred_mean = torch.tensor([[0.02776188589632511139],
        [0.02776273339986801147],
        [0.02774043381214141846],
        [0.02774744480848312378],
        [0.02774233743548393250],
        [0.02777685970067977905],
        [0.02774161472916603088],
        [0.02775483392179012299],
        [0.02773117274045944214],
        [0.02773284725844860077],
        [0.02769741788506507874],
        [0.02767934836447238922],
        [0.02776344865560531616],
        [0.02775776386260986328],
        [0.02774072624742984772]], device='cuda:0')
pred_mean = pred_mean.cpu()
pred_std = np.array([[1.6918170e-08],
 [1.6850935e-08],
 [1.6184089e-08],
 [1.6973054e-08],
 [1.6691770e-08],
 [1.6389905e-08],
 [1.6254067e-08],
 [1.6244462e-08],
 [1.6704119e-08],
 [1.6444790e-08],
 [1.7047148e-08],
 [1.6344625e-08],
 [1.6831725e-08],
 [1.6745282e-08],
 [1.6768610e-08],
 [1.6754887e-08]])

#pred_std = torch.tensor(pred_std)


plt.figure(figsize=(12, 6))

# 子图 1：pred_mean 的变化
plt.subplot(1, 2, 1)
plt.plot(pred_mean, label="Predicted Mean", color="blue")
plt.xlabel("Batch Index")
plt.ylabel("Predicted Confidence")
plt.title("Predicted Mean Over Batches")
plt.legend()
plt.grid(True)

# 子图 2：pred_std 的变化

plt.subplot(1, 2, 2)
plt.plot(pred_std, label="Predicted Std", color="red")
plt.xlabel("Batch Index")
plt.ylabel("Uncertainty (Std)")
plt.title("Predicted Uncertainty Over Batches")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()



# 将 pred_mean 转换为 numpy 数组并展平
pred_mean = pred_mean.cpu().numpy().flatten()

# 将 pred_std 展平
#pred_std = pred_std.flatten()
pred_std = pred_std.flatten()[:len(pred_mean)]

# 创建 x 轴数据（批次索引）
batch_indices = np.arange(len(pred_mean))

# 绘制误差条图
plt.figure(figsize=(10, 6))

plt.errorbar(batch_indices, pred_mean, yerr=pred_std, fmt='o', color='blue', ecolor='lightblue', capsize=5, label='Predicted Mean ± Std')

# 设置图形属性
plt.xlabel("Batch Index")
plt.ylabel("Predicted Confidence")
plt.title("Predicted Confidence with Error Bars")
plt.legend()
plt.grid(True)
plt.show()
