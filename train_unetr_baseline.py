import os
import glob
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# 💡 引入 MONAI 中经典的纯视觉 Transformer 架构
from monai.networks.nets import UNETR
from torch.amp import autocast, GradScaler

# ==========================================
# 1. 数据集读取器 (绝对公平原则：代码 100% 不变)
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
	# --- ⚙️ 路径配置 ---
	DIRTY_TRAIN = r"G:\CT_RE\dataset\Final_Adaptive_Dirty\train"
	CLEAN_TRAIN = r"G:\CT_RE\dataset\Final_Adaptive_GT\train"
	DIRTY_TEST  = r"G:\CT_RE\dataset\Final_Adaptive_Dirty\test"
	CLEAN_TEST  = r"G:\CT_RE\dataset\Final_Adaptive_GT\test"

	# 🚨 警告：纯 ViT 极其消耗显存！
	# 强烈建议保持 BATCH_SIZE = 1，并利用梯度累加(这里暂未加入)来模拟大 Batch。
	BATCH_SIZE = 1
	EPOCHS = 100
	LR = 1e-4
	SAVE_DIR = "checkpoints"
	os.makedirs(SAVE_DIR, exist_ok=True)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"🔥 启动对比基线实验 | 设备: {device} | 架构: 纯 Transformer (UNETR)")

	# --- 📦 加载数据 ---
	train_dataset = CLArtifactDataset(DIRTY_TRAIN, CLEAN_TRAIN)
	test_dataset = CLArtifactDataset(DIRTY_TEST, CLEAN_TEST)
	train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
	test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

	# ==========================================================
	# 🧠 构建标准的 UNETR 模型 (基于经典 ViT)
	# ==========================================================
	model = UNETR(
		spatial_dims=3,          # 💡 修复 1：显式声明这是 3D 物理空间张量
		in_channels=1,
		out_channels=1,
		img_size=(64, 256, 256), # 💡 注意：这里必须是你 Padding 之后的真实送入尺寸！
		feature_size=16,         # 适当调低特征数以保住你的 24G 显存
		hidden_size=768,         # ViT-Base 的标准隐藏层维度
		mlp_dim=3072,            # ViT-Base 的标准前馈层维度
		num_heads=12,            # 12 个自注意力头
		# pos_embed="perceptron",  <--- ❌ 修复 2：删掉这一行！让系统自动使用默认的绝对位置编码
		norm_name="instance",
		res_block=True
	).to(device)

	# --- 🎯 损失函数与优化器 (保持绝对公平) ---
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

			# Z 轴物理镜像补齐 (40层 -> 64层)
			# 这一步刚好对应了上方模型初始化时写死的 img_size=(64, 256, 256)
			dirty_padded = F.pad(dirty, (0, 0, 0, 0, 12, 12), mode='reflect')

			optimizer.zero_grad()

			# 使用自动混合精度 (AMP) 极其重要，否则纯 ViT 的 QKV 矩阵乘法会慢到你怀疑人生
			with autocast('cuda'):
				preds_padded = model(dirty_padded)
				# 裁剪对齐，计算真实的 40 层 Loss
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

		# 💡 保存为纯 Transformer 权重
		if val_loss < best_val_loss:
			best_val_loss = val_loss
			best_weight_path = os.path.join(SAVE_DIR, "unetr_baseline_best2.pth")
			torch.save(model.state_dict(), best_weight_path)
			print(f"   🌟 [纯 Transformer] 验证集 Loss 刷新纪录！已保存至 -> {best_weight_path}")

	print("\n🎉 纯 Transformer 基线模型训练完毕！")

if __name__ == "__main__":
	torch.multiprocessing.freeze_support()
	main()