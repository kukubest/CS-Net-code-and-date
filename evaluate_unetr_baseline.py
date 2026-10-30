import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from monai.networks.nets import UNETR  # 💡 引入纯 Transformer
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error

# ==========================================
# 1. 数据集读取器 (绝对公平，与前两个模型完全一致)
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

		# 兼容读取 float16 或 float32
		def safe_load(path):
			file_size = os.path.getsize(path)
			if file_size == np.prod(self.shape) * 2:
				return np.fromfile(path, dtype=np.float16).astype(np.float32).reshape(self.shape)
			return np.fromfile(path, dtype=np.float32).reshape(self.shape)

		dirty_vol = safe_load(dirty_path)
		clean_vol = safe_load(clean_path)

		return torch.from_numpy(dirty_vol).unsqueeze(0), torch.from_numpy(clean_vol).unsqueeze(0), filename

# ==========================================
# 2. 核心评估引擎
# ==========================================
def main():
	# --- ⚙️ 路径配置 ---
	DIRTY_TEST  = r"G:\CT_RE\dataset\Final_Adaptive_Dirty\test"
	CLEAN_TEST  = r"G:\CT_RE\dataset\Final_Adaptive_GT\test"
	WEIGHT_PATH = r"checkpoints\unetr_baseline_best.pth" # 💡 指向纯 Transformer 权重

	# 预测结果保存目录
	SAVE_DIR = r"G:\CT_RE\dataset\Final_Predictions"
	os.makedirs(SAVE_DIR, exist_ok=True)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"🚀 启动基线模型评估引擎 | 设备: {device} | 架构: 纯 Transformer (UNETR)")

	# --- 📦 加载数据 ---
	test_dataset = CLArtifactDataset(DIRTY_TEST, CLEAN_TEST)
	test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

	# --- 🧠 加载 UNETR 架构与权重 ---
	print(f"📥 正在挂载基线权重: {WEIGHT_PATH}")
	model = UNETR(
		spatial_dims=3,          # 声明为 3D 物理空间
		in_channels=1,
		out_channels=1,
		img_size=(64, 256, 256), # 必须匹配 Padding 后的输入尺寸
		feature_size=16,         # 必须匹配训练时的配置
		hidden_size=768,
		mlp_dim=3072,
		num_heads=12,
		norm_name="instance",
		res_block=True
	).to(device)

	model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device, weights_only=True))
	model.eval()

	# --- 📊 性能记录指标 ---
	metrics = {'dirty_rmse': [], 'dirty_ssim': [], 'dirty_psnr': [],
			   'pred_rmse': [],  'pred_ssim': [],  'pred_psnr': []}

	print("\n" + "="*50)
	print(">>> 开始进行纯 Transformer (UNETR) 跑分测试 <<<")
	print("="*50)

	with torch.no_grad():
		for i, (dirty, clean, filename) in enumerate(test_loader):
			filename = filename[0]
			dirty, clean = dirty.to(device), clean.to(device)

			# 💡 物理镜像补齐 (严格保证测试条件的公平性 40 -> 64)
			dirty_padded = F.pad(dirty, (0, 0, 0, 0, 12, 12), mode='reflect')

			# 🚀 模型推断
			with torch.amp.autocast('cuda'):
				preds_padded = model(dirty_padded)

			# 💡 截断提取 (64 -> 40)，拿回真实厚度
			preds_aligned = preds_padded[:, :, 12:52, :, :]

			# 转移回 CPU 并转换为 Numpy
			np_dirty = dirty.squeeze().cpu().numpy()
			np_clean = clean.squeeze().cpu().numpy()
			np_pred = preds_aligned.squeeze().cpu().numpy()

			# 限制输出范围
			np_pred = np.clip(np_pred, 0.0, 1.0)

			# --- 📈 计算量化指标 ---
			rmse_dirty = np.sqrt(mean_squared_error(np_clean, np_dirty))
			rmse_pred = np.sqrt(mean_squared_error(np_clean, np_pred))

			psnr_dirty = peak_signal_noise_ratio(np_clean, np_dirty, data_range=1.0)
			psnr_pred = peak_signal_noise_ratio(np_clean, np_pred, data_range=1.0)

			ssim_dirty = structural_similarity(np_clean, np_dirty, data_range=1.0)
			ssim_pred = structural_similarity(np_clean, np_pred, data_range=1.0)

			metrics['dirty_rmse'].append(rmse_dirty)
			metrics['pred_rmse'].append(rmse_pred)
			metrics['dirty_psnr'].append(psnr_dirty)
			metrics['pred_psnr'].append(psnr_pred)
			metrics['dirty_ssim'].append(ssim_dirty)
			metrics['pred_ssim'].append(ssim_pred)

			# --- 💾 保存 UNETR 的预测结果 ---
			# 命名为 _unetr_pred.raw
			save_name = filename.replace("_dirty.raw", "_unetr_pred.raw")
			np_pred.tofile(os.path.join(SAVE_DIR, save_name))

			if (i+1) % 20 == 0 or i == 0:
				print(f"  [{i+1}/{len(test_dataset)}] {filename} | "
					  f"UNETR PSNR: {psnr_pred:.2f} dB | SSIM: {ssim_pred:.4f}")

	# --- 🏆 汇总报告 ---
	print("\n" + "🌟"*25)
	print(" "*8 + "纯 Transformer (UNETR) 最终测试集平均性能报告")
	print("🌟"*25)

	print(f"【RMSE (均方根误差)】")
	print(f"  > 原始 CL 脏图: {np.mean(metrics['dirty_rmse']):.4f}")
	print(f"  > UNETR 去伪影: {np.mean(metrics['pred_rmse']):.4f}")

	print(f"\n【PSNR (峰值信噪比)】")
	print(f"  > 原始 CL 脏图: {np.mean(metrics['dirty_psnr']):.2f} dB")
	print(f"  > UNETR 去伪影: {np.mean(metrics['pred_psnr']):.2f} dB")

	print(f"\n【SSIM (结构相似度)】")
	print(f"  > 原始 CL 脏图: {np.mean(metrics['dirty_ssim']):.4f}")
	print(f"  > UNETR 去伪影: {np.mean(metrics['pred_ssim']):.4f}")
	print("🌟"*25)
	print(f"💾 所有 UNETR 预测结果已保存，后缀为 _unetr_pred.raw")

if __name__ == "__main__":
	main()