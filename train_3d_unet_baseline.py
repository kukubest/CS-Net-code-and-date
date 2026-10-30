import os
import glob
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from monai.networks.nets import UNet # 💡 引入经典的 U-Net
from torch.amp import autocast, GradScaler

# ==========================================
# 1. 数据集读取器 (与 Swin-UNETR 保持 100% 相同)
# ==========================================
class CLArtifactDataset(Dataset):
	def __init__(self, dirty_dir, clean_dir, shape=(40, 256, 256)):
		self.shape = shape
		self.dirty_files = sorted(glob.glob(os.path.join(dirty_dir, "*.raw")))
		self.clean_dir = clean_dir

		if len(self.dirty_files) == 0:
			raise ValueError(f"❌ 未找到任何数据！")

	def __len__(self):
		return len(self.dirty_files)

	def __getitem__(self, idx):
		dirty_path = self.dirty_files[idx]
		filename = os.path.basename(dirty_path)
		clean_name = filename.replace("_dirty.raw", ".raw")
		clean_path = os.path.join(self.clean_dir, clean_name)

		dirty_vol = np.fromfile(dirty_path, dtype=np.float32).reshape(self.shape)
		clean_vol = np.fromfile(clean_path, dtype=np.float32).reshape(self.shape)

		dirty_tensor = torch.from_numpy(dirty_vol).unsqueeze(0)
		clean_tensor = torch.from_numpy(clean_vol).unsqueeze(0)

		return dirty_tensor, clean_tensor

# ==========================================
# 2. 核心训练引擎
# ==========================================
def main():
	# --- ⚙️ 路径配置 (务必与之前完全一致) ---
	DIRTY_TRAIN = r"G:\CT_RE\dataset\Final_Adaptive_Dirty\train"
	CLEAN_TRAIN = r"G:\CT_RE\dataset\Final_Adaptive_GT\train"
	DIRTY_TEST  = r"G:\CT_RE\dataset\Final_Adaptive_Dirty\test"
	CLEAN_TEST  = r"G:\CT_RE\dataset\Final_Adaptive_GT\test"

	# 因为 3D U-Net 显存开销小，3090Ti 完全可以把 Batch Size 开到 4 甚至 8 以加速训练
	BATCH_SIZE = 4
	EPOCHS = 100
	LR = 1e-4
	SAVE_DIR = "checkpoints"
	os.makedirs(SAVE_DIR, exist_ok=True)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"🔥 启动对比基线实验 | 设备: {device} | 架构: 3D U-Net")

	# --- 📦 加载数据 ---
	train_dataset = CLArtifactDataset(DIRTY_TRAIN, CLEAN_TRAIN)
	test_dataset = CLArtifactDataset(DIRTY_TEST, CLEAN_TEST)
	train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
	test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

	# ==========================================================
	# 🧠 构建经典的 3D U-Net 模型 (带残差连接，使其具备较强竞争力)
	# ==========================================================
	model = UNet(
		spatial_dims=3,
		in_channels=1,
		out_channels=1,
		channels=(16, 32, 64, 128, 256), # 从浅到深的特征图通道数
		strides=(2, 2, 2, 2),            # 4层下采样，空间缩小 16 倍 (依然需要整除验证)
		num_res_units=2,                 # 引入残差单元 (Residual Units) 防止梯度消失
		norm="instance"                  # 实例归一化，在医疗/工业 3D 图像中表现最好
	).to(device)

	# --- 🎯 损失函数与优化器 ---
	criterion = nn.L1Loss()
	optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
	scaler = GradScaler('cuda')

	# --- 🚀 开始训练 ---
	best_val_loss = float('inf')

	for epoch in range(1, EPOCHS + 1):
		start_time = time.time()
		model.train()
		train_loss = 0.0

		for batch_idx, (dirty, clean) in enumerate(train_loader):
			dirty, clean = dirty.to(device), clean.to(device)

			# 保持一致：Z 轴物理镜像补齐 (40层 -> 64层)
			# 传统 3D U-Net 同样需要长宽深能被 strides 的乘积整除，64 完美兼容。
			dirty_padded = F.pad(dirty, (0, 0, 0, 0, 12, 12), mode='reflect')

			optimizer.zero_grad()

			with autocast('cuda'):
				preds_padded = model(dirty_padded)
				preds_aligned = preds_padded[:, :, 12:52, :, :]
				loss = criterion(preds_aligned, clean)

			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()

			train_loss += loss.item() * dirty.size(0)

		train_loss /= len(train_loader.dataset)
		scheduler.step()

		# --- 🔍 验证阶段 ---
		model.eval()
		val_loss = 0.0
		with torch.no_grad():
			for dirty, clean in test_loader:
				dirty, clean = dirty.to(device), clean.to(device)

				dirty_padded = F.pad(dirty, (0, 0, 0, 0, 12, 12), mode='reflect')

				with autocast('cuda'):
					preds_padded = model(dirty_padded)
					preds_aligned = preds_padded[:, :, 12:52, :, :]
					loss = criterion(preds_aligned, clean)

				val_loss += loss.item() * dirty.size(0)

		val_loss /= len(test_loader.dataset)
		epoch_time = time.time() - start_time

		print(f"Epoch [{epoch:03d}/{EPOCHS}] | Time: {epoch_time:.0f}s | LR: {scheduler.get_last_lr()[0]:.2e} | "
			  f"Train: {train_loss:.5f} | Val: {val_loss:.5f}")

		# 💡 保存时修改文件名为 3d_unet_best.pth，防止覆盖你的 Swin-UNETR 权重！
		if val_loss < best_val_loss:
			best_val_loss = val_loss
			best_weight_path = os.path.join(SAVE_DIR, "3d_unet_best.pth")
			torch.save(model.state_dict(), best_weight_path)
			print(f"   🌟 [3D U-Net] 验证集 Loss 刷新纪录！已保存至 -> {best_weight_path}")

	print("\n🎉 基线模型 3D U-Net 训练完毕！你可以使用相同的评估脚本进行跑分对比了。")

if __name__ == "__main__":
	torch.multiprocessing.freeze_support()
	main()