import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from monai.networks.nets import SwinUNETR
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error

# ==========================================
# 1. 终极混合网络架构定义 (必须与训练时完全一致才能加载权重)
# ==========================================
class FastRobustCLNet(nn.Module):
	def __init__(self, img_size=(64, 256, 256), in_channels=1, out_channels=1, feature_size=24):
		super().__init__()

		self.stem_enc = nn.Sequential(
			nn.Conv3d(in_channels, feature_size, kernel_size=3, stride=1, padding=1),
			nn.InstanceNorm3d(feature_size),
			nn.LeakyReLU(inplace=True)
		)

		self.downsample = nn.Sequential(
			nn.Conv3d(feature_size, feature_size, kernel_size=3, stride=2, padding=1),
			nn.InstanceNorm3d(feature_size),
			nn.LeakyReLU(inplace=True)
		)

		# 显式声明 3D 空间，动态适应尺寸
		self.backbone = SwinUNETR(
			spatial_dims=3,
			in_channels=feature_size,
			out_channels=feature_size,
			feature_size=feature_size,
			use_checkpoint=False # 评估时不需要检查点，全速前进
		)

		self.upsample = nn.Sequential(
			nn.ConvTranspose3d(feature_size, feature_size, kernel_size=2, stride=2),
			nn.InstanceNorm3d(feature_size),
			nn.LeakyReLU(inplace=True)
		)

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
		fusion = torch.cat([x_up, x_high_res], dim=1)
		out = self.tail_dec(fusion)
		return out

# ==========================================
# 2. 数据集读取器 (绝对公平对齐)
# ==========================================
class CLArtifactDataset(Dataset):
	def __init__(self, dirty_dir, clean_dir, shape=(40, 256, 256)):
		self.shape = shape
		self.dirty_files = sorted(glob.glob(os.path.join(dirty_dir, "*.raw")))
		self.clean_dir = clean_dir

	def __len__(self):
		return len(self.dirty_files)

	def __getitem__(self, idx):
		dirty_path = self.dirty_files[idx]
		filename = os.path.basename(dirty_path)
		clean_name = filename.replace("_dirty.raw", ".raw")
		clean_path = os.path.join(self.clean_dir, clean_name)

		def safe_load(path):
			file_size = os.path.getsize(path)
			if file_size == np.prod(self.shape) * 2:
				return np.fromfile(path, dtype=np.float16).astype(np.float32).reshape(self.shape)
			return np.fromfile(path, dtype=np.float32).reshape(self.shape)

		dirty_vol = safe_load(dirty_path)
		clean_vol = safe_load(clean_path)

		return torch.from_numpy(dirty_vol).unsqueeze(0), torch.from_numpy(clean_vol).unsqueeze(0), filename

# ==========================================
# 3. 核心评估引擎
# ==========================================
def main():
	# --- ⚙️ 路径配置 ---
	DIRTY_TEST  = r"G:\CT_RE\dataset\Final_Adaptive_Dirty\test"
	CLEAN_TEST  = r"G:\CT_RE\dataset\Final_Adaptive_GT\test"
	WEIGHT_PATH = r"checkpoints\fast_robust_clnet_best.pth" # 💡 指向最新混合架构权重

	SAVE_DIR = r"G:\CT_RE\dataset\Final_Predictions"
	os.makedirs(SAVE_DIR, exist_ok=True)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"🚀 启动终极混合架构评估引擎 | 设备: {device}")

	test_dataset = CLArtifactDataset(DIRTY_TEST, CLEAN_TEST)
	test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

	# --- 🧠 加载模型与权重 ---
	print(f"📥 正在挂载权重: {WEIGHT_PATH}")
	model = FastRobustCLNet(img_size=(64, 256, 256), in_channels=1, out_channels=1, feature_size=24).to(device)

	# # 尝试开启算子融合以获得极限推理速度
	# try:
	# 	model = torch.compile(model, mode="reduce-overhead")
	# except:
	# 	pass # Windows 环境正常回退

	model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device, weights_only=True))
	model.eval()

	# --- 📊 性能记录指标 ---
	metrics = {'dirty_rmse': [], 'dirty_ssim': [], 'dirty_psnr': [],
			   'pred_rmse': [],  'pred_ssim': [],  'pred_psnr': []}

	print("\n" + "="*50)
	print(">>> 开始进行 FastRobustCLNet 跑分测试 <<<")
	print("="*50)

	import time
	total_infer_time = 0.0

	with torch.no_grad():
		for i, (dirty, clean, filename) in enumerate(test_loader):
			filename = filename[0]
			dirty, clean = dirty.to(device), clean.to(device)

			# 物理镜像补齐
			dirty_padded = F.pad(dirty, (0, 0, 0, 0, 12, 12), mode='reflect')

			# 🚀 模型推断与测速
			torch.cuda.synchronize()
			t0 = time.time()

			with torch.amp.autocast('cuda'):
				preds_padded = model(dirty_padded)

			torch.cuda.synchronize()
			total_infer_time += (time.time() - t0)

			# 截断提取
			preds_aligned = preds_padded[:, :, 12:52, :, :]

			# 转移回 CPU 并转换为 Numpy
			np_dirty = dirty.squeeze().cpu().numpy()
			np_clean = clean.squeeze().cpu().numpy()
			np_pred = preds_aligned.squeeze().cpu().numpy()

			np_pred = np.clip(np_pred, 0.0, 1.0)

			# --- 📈 计算量化指标 ---
			rmse_pred = np.sqrt(mean_squared_error(np_clean, np_pred))
			psnr_pred = peak_signal_noise_ratio(np_clean, np_pred, data_range=1.0)
			ssim_pred = structural_similarity(np_clean, np_pred, data_range=1.0)

			metrics['pred_rmse'].append(rmse_pred)
			metrics['pred_psnr'].append(psnr_pred)
			metrics['pred_ssim'].append(ssim_pred)

			# --- 💾 保存预测结果 ---
			save_name = filename.replace("_dirty.raw", "_fast_robust_pred.raw")
			np_pred.tofile(os.path.join(SAVE_DIR, save_name))

			if (i+1) % 20 == 0 or i == 0:
				print(f"  [{i+1}/{len(test_dataset)}] {filename} | "
					  f"PSNR: {psnr_pred:.2f} dB | SSIM: {ssim_pred:.4f}")

	# --- 🏆 汇总报告 ---
	print("\n" + "🌟"*25)
	print(" "*4 + "FastRobustCLNet (终极混合架构) 最终性能报告")
	print("🌟"*25)

	print(f"【量化精度指标】")
	print(f"  > 平均 RMSE: {np.mean(metrics['pred_rmse']):.4f}")
	print(f"  > 平均 PSNR: {np.mean(metrics['pred_psnr']):.2f} dB")
	print(f"  > 平均 SSIM: {np.mean(metrics['pred_ssim']):.4f}")

	avg_ms = (total_infer_time / len(test_dataset)) * 1000
	print(f"\n【工程部署指标】")
	print(f"  > 测试集单块平均推理耗时: {avg_ms:.2f} ms")

	print("🌟"*25)
	print(f"💾 预测结果已保存，后缀为 _fast_robust_pred.raw")

if __name__ == "__main__":
	main()