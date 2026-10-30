import os
import glob
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # 用于 3D 镜像 Padding
from torch.utils.data import Dataset, DataLoader
from monai.networks.nets import SwinUNETR
from torch.amp import autocast, GradScaler

# ==========================================
# 1. 自定义数据集读取器 (Dataset)
# ==========================================
class CLArtifactDataset(Dataset):
	def __init__(self, dirty_dir, clean_dir, shape=(40, 256, 256)):
		self.shape = shape
		self.dirty_files = sorted(glob.glob(os.path.join(dirty_dir, "*.raw")))
		self.clean_dir = clean_dir

		if len(self.dirty_files) == 0:
			raise ValueError(f"❌ 在 {dirty_dir} 中未找到任何数据！")

	def __len__(self):
		return len(self.dirty_files)

	def __getitem__(self, idx):
		dirty_path = self.dirty_files[idx]
		filename = os.path.basename(dirty_path)

		# 将文件名从 "_dirty.raw" 替换回 ".raw" 寻找标签
		clean_name = filename.replace("_dirty.raw", ".raw")
		clean_path = os.path.join(self.clean_dir, clean_name)

		# 读取 3D 矩阵
		dirty_vol = np.fromfile(dirty_path, dtype=np.float32).reshape(self.shape)
		clean_vol = np.fromfile(clean_path, dtype=np.float32).reshape(self.shape)

		# 增加 Channel 维度 -> 最终形状 [1, 40, 256, 256]
		dirty_tensor = torch.from_numpy(dirty_vol).unsqueeze(0)
		clean_tensor = torch.from_numpy(clean_vol).unsqueeze(0)

		return dirty_tensor, clean_tensor

# ==========================================
# 2. 核心训练引擎
# ==========================================
def main():
	# --- ⚙️ 路径与超参数配置 ---
	# 请确保路径与你生成的最终数据集一致
	DIRTY_TRAIN = r"G:\CT_RE\dataset\Final_Adaptive_Dirty\train"
	CLEAN_TRAIN = r"G:\CT_RE\dataset\Final_Adaptive_GT\train"
	DIRTY_TEST  = r"G:\CT_RE\dataset\Final_Adaptive_Dirty\test"
	CLEAN_TEST  = r"G:\CT_RE\dataset\Final_Adaptive_GT\test"

	BATCH_SIZE = 2      # RTX 3090Ti 24G 跑 64x256x256 建议保持 2 (如果显存溢出可改为 1)
	EPOCHS = 100        # 总训练轮数
	LR = 1e-4           # 初始学习率
	SAVE_DIR = "checkpoints"
	os.makedirs(SAVE_DIR, exist_ok=True)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"🔥 正在使用设备: {device} | AMP混合精度开启")

	# --- 📦 数据加载 ---
	print("📥 正在挂载数据集...")
	train_dataset = CLArtifactDataset(DIRTY_TRAIN, CLEAN_TRAIN)
	test_dataset = CLArtifactDataset(DIRTY_TEST, CLEAN_TEST)

	# Windows 环境下 num_workers 不要开太大，2 或 4 即可
	train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
	test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
	print(f"✅ 训练集样本: {len(train_dataset)} | 测试集样本: {len(test_dataset)}")

	# --- 🧠 构建 3D Swin-UNETR 模型 ---
	# 注意：此时模型认为它要处理的是 64 的整数倍
	model = SwinUNETR(
		spatial_dims=3,
		in_channels=1,
		out_channels=1,
		feature_size=24,   # 基础特征通道数
		use_checkpoint=True # 开启梯度检查点，用一点点速度换取近乎一半的显存节约！
	).to(device)

	# --- 🎯 损失函数与优化器 ---
	# 使用 L1 Loss 能够更好地保留 PCB 线路和打孔的锐利高频边缘
	criterion = nn.L1Loss()
	optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

	# 混合精度梯度缩放器
	scaler = GradScaler('cuda')

	# --- 🚀 开始训练循环 ---
	best_val_loss = float('inf')

	for epoch in range(1, EPOCHS + 1):
		start_time = time.time()
		model.train()
		train_loss = 0.0

		for batch_idx, (dirty, clean) in enumerate(train_loader):
			# 将数据转移到 GPU
			dirty, clean = dirty.to(device), clean.to(device)

			# ==========================================================
			# 💡 核心工程策略 1：Z 轴物理镜像补齐 (40层 -> 64层)
			# 为了满足 Swin-UNETR 的 32 倍数要求，我们利用 mode='reflect'
			# 从原有的 40 层两端，像照镜子一样各镜像延伸出 12 层。
			# F.pad 的顺序是 (X左, X右, Y上, Y下, Z前, Z后)
			# ==========================================================
			dirty_padded = F.pad(dirty, (0, 0, 0, 0, 12, 12), mode='reflect')

			optimizer.zero_grad()

			# 使用自动混合精度 (AMP) 进行前向传播
			with autocast('cuda'):
				# 喂入 64 层的 padded 数据，模型吐出 64 层的预测结果
				preds_padded = model(dirty_padded)

				# ==========================================================
				# 💡 核心工程策略 2：截断提取与维度对齐
				# 我们切掉预测结果中多余的上下各 12 层“假皮”，只提取中间真实的 40 层
				# ==========================================================
				preds_aligned = preds_padded[:, :, 12:52, :, :]

				# 在真实物理维度的 40 层上计算 Loss，绝不浪费算力去拟合边缘
				loss = criterion(preds_aligned, clean)

			# 缩放梯度，反向传播
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()

			train_loss += loss.item() * dirty.size(0)

		train_loss /= len(train_loader.dataset)
		scheduler.step() # 更新学习率

		# --- 🔍 验证阶段 (验证集跑分) ---
		model.eval()
		val_loss = 0.0
		with torch.no_grad():
			for dirty, clean in test_loader:
				dirty, clean = dirty.to(device), clean.to(device)

				# 验证集同样执行“镜像补齐 -> 预测 -> 截断 -> 算Loss”的闭环
				dirty_padded = F.pad(dirty, (0, 0, 0, 0, 12, 12), mode='reflect')

				with autocast('cuda'):
					preds_padded = model(dirty_padded)
					preds_aligned = preds_padded[:, :, 12:52, :, :]
					loss = criterion(preds_aligned, clean)

				val_loss += loss.item() * dirty.size(0)

		val_loss /= len(test_loader.dataset)
		epoch_time = time.time() - start_time

		# 打印日志
		print(f"Epoch [{epoch:03d}/{EPOCHS}] | Time: {epoch_time:.0f}s | LR: {scheduler.get_last_lr()[0]:.2e} | "
			  f"Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

		# --- 💾 保存最强王者权重 ---
		if val_loss < best_val_loss:
			best_val_loss = val_loss
			best_weight_path = os.path.join(SAVE_DIR, "swin_unetr_best.pth")
			torch.save(model.state_dict(), best_weight_path)
			print(f"   🌟 验证集泛化误差创历史新低！已保存最佳模型 -> {best_weight_path}")

	print("\n🎉 训练圆满结束！你的工业级 CL 去伪影大模型已就绪！")

if __name__ == "__main__":
	# 防止 Windows 下多线程 DataLoader 崩溃的保护机制
	torch.multiprocessing.freeze_support()
	main()