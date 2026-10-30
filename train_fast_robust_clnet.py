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
# 1. 终极混合网络架构定义 (FastRobustCLNet)
# ==========================================
class FastRobustCLNet(nn.Module):
	"""
	极速抗噪混合架构 (Fast & Robust CNN-Swin Architecture)
	通过【下采样前置】极大降低 FLOPs，并通过【宏观跳跃连接】保护极高频工业边缘
	"""
	def __init__(self, img_size=(64, 256, 256), in_channels=1, out_channels=1, feature_size=24):
		super().__init__()

		# 1. 高频特征提取与抗噪网关 (保持原分辨率)
		self.stem_enc = nn.Sequential(
			nn.Conv3d(in_channels, feature_size, kernel_size=3, stride=1, padding=1),
			nn.InstanceNorm3d(feature_size),
			nn.LeakyReLU(inplace=True)
		)

		# 2. 空间降维层 (体积缩小 8 倍)
		self.downsample = nn.Sequential(
			nn.Conv3d(feature_size, feature_size, kernel_size=3, stride=2, padding=1),
			nn.InstanceNorm3d(feature_size),
			nn.LeakyReLU(inplace=True)
		)

		downsampled_size = (img_size[0] // 2, img_size[1] // 2, img_size[2] // 2)

		# 3. 核心推理引擎：Swin-UNETR (低分辨率隐空间极速运行)
		self.backbone = SwinUNETR(
			spatial_dims=3,              # ✅ 显式声明 3D 空间即可
			in_channels=feature_size,
			out_channels=feature_size,
			feature_size=feature_size,
			use_checkpoint=True
		)

		# 4. 空间升维层 (还原回原始物理空间)
		self.upsample = nn.Sequential(
			nn.ConvTranspose3d(feature_size, feature_size, kernel_size=2, stride=2),
			nn.InstanceNorm3d(feature_size),
			nn.LeakyReLU(inplace=True)
		)

		# 5. 宏观解码器 (接收跳跃连接，还原锐利边缘)
		self.tail_dec = nn.Sequential(
			nn.Conv3d(feature_size * 2, feature_size, kernel_size=3, stride=1, padding=1),
			nn.InstanceNorm3d(feature_size),
			nn.LeakyReLU(inplace=True),
			nn.Conv3d(feature_size, out_channels, kernel_size=3, stride=1, padding=1)
		)

	def forward(self, x):
		x_high_res = self.stem_enc(x)
		x_down = self.downsample(x_high_res)
		swin_out = self.backbone(x_down)
		x_up = self.upsample(swin_out)
		fusion = torch.cat([x_up, x_high_res], dim=1) # 融合高频边缘与低频宏观拓扑
		out = self.tail_dec(fusion)
		return out

# ==========================================
# 2. 自定义数据集读取器 (Dataset) - 保持不变
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
		clean_name = filename.replace("_dirty.raw", ".raw")
		clean_path = os.path.join(self.clean_dir, clean_name)

		dirty_vol = np.fromfile(dirty_path, dtype=np.float32).reshape(self.shape)
		clean_vol = np.fromfile(clean_path, dtype=np.float32).reshape(self.shape)

		dirty_tensor = torch.from_numpy(dirty_vol).unsqueeze(0)
		clean_tensor = torch.from_numpy(clean_vol).unsqueeze(0)

		return dirty_tensor, clean_tensor

# ==========================================
# 3. 核心训练引擎
# ==========================================
def main():
	DIRTY_TRAIN = r"G:\CT_RE\dataset\Final_Adaptive_Dirty\train"
	CLEAN_TRAIN = r"G:\CT_RE\dataset\Final_Adaptive_GT\train"
	DIRTY_TEST  = r"G:\CT_RE\dataset\Final_Adaptive_Dirty\test"
	CLEAN_TEST  = r"G:\CT_RE\dataset\Final_Adaptive_GT\test"

	# 💡 显存红利：由于引入了前置下采样，这里的 BATCH_SIZE 极有可能可以开到 4 甚至 6！
	# 建议你先用 2 跑通，再慢慢往上调，榨干 3090Ti 的最后一滴显存。
	BATCH_SIZE = 2
	EPOCHS = 100
	LR = 1e-4
	SAVE_DIR = "checkpoints"
	os.makedirs(SAVE_DIR, exist_ok=True)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"🔥 正在使用设备: {device} | AMP混合精度开启")

	print("📥 正在挂载数据集...")
	train_dataset = CLArtifactDataset(DIRTY_TRAIN, CLEAN_TRAIN)
	test_dataset = CLArtifactDataset(DIRTY_TEST, CLEAN_TEST)

	train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
	test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

	# --- 🧠 挂载最新的混合大模型 ---
	print("🚀 初始化 FastRobustCLNet (三明治混合架构)...")
	model = FastRobustCLNet(
		img_size=(64, 256, 256), # 必须匹配 padding 后的尺寸
		in_channels=1,
		out_channels=1,
		feature_size=24
	).to(device)

	# # --- ⚡ 软件级白嫖提速 (算子融合) ---
	# try:
	# 	print("⚡ 尝试启动 torch.compile 底层算子融合...")
	# 	model = torch.compile(model, mode="reduce-overhead")
	# 	print("✅ 算子融合就绪！初次 Forward 可能需要 1-2 分钟编译，请耐心等待。")
	# except Exception as e:
	# 	print(f"⚠️ 当前环境暂不支持 torch.compile (Windows常见)，已回退常规模式运行。")

	criterion = nn.L1Loss()
	optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
	scaler = GradScaler('cuda')

	best_val_loss = float('inf')

	for epoch in range(1, EPOCHS + 1):
		start_time = time.time()
		model.train()
		train_loss = 0.0

		for batch_idx, (dirty, clean) in enumerate(train_loader):
			dirty, clean = dirty.to(device), clean.to(device)

			# Z 轴物理镜像补齐 (40层 -> 64层)
			dirty_padded = F.pad(dirty, (0, 0, 0, 0, 12, 12), mode='reflect')

			optimizer.zero_grad()

			with autocast('cuda'):
				# 喂入极速混合架构
				preds_padded = model(dirty_padded)

				# 截断提取中间真实的 40 层
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
			  f"Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

		# --- 💾 保存最终架构的最强权重 ---
		if val_loss < best_val_loss:
			best_val_loss = val_loss
			# 💡 更改了保存名称，避免覆盖之前的纯 Swin 权重，方便你做消融对比
			best_weight_path = os.path.join(SAVE_DIR, "fast_robust_clnet_best.pth")
			torch.save(model.state_dict(), best_weight_path)
			print(f"   🌟 混合架构创历史新低！已保存 -> {best_weight_path}")

	print("\n🎉 训练圆满结束！你的工业级极速 CL 去伪影大模型已就绪！")

if __name__ == "__main__":
	import os
	# 防止 Windows/Intel 平台下 OMP 冲突导致的崩溃
	os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
	torch.multiprocessing.freeze_support()
	main()