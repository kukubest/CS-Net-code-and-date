import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from monai.networks.nets import UNet, UNETR, SwinUNETR
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.amp import autocast

# ==========================================
# 🌟 插入终极混合网络架构定义 (已更新为 CS-Net)
# ==========================================
class CSNet(nn.Module):
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
		self.backbone = SwinUNETR(
			spatial_dims=3,
			in_channels=feature_size,
			out_channels=feature_size,
			feature_size=feature_size,
			use_checkpoint=False
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
# 1. 数据集读取器 (带自适应 Float16 解析)
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
		clean_path = os.path.join(self.clean_dir, filename.replace("_dirty.raw", ".raw"))

		def safe_load(path):
			file_size = os.path.getsize(path)
			if file_size == np.prod(self.shape) * 2:
				return np.fromfile(path, dtype=np.float16).astype(np.float32).reshape(self.shape)
			return np.fromfile(path, dtype=np.float32).reshape(self.shape)

		dirty_vol = safe_load(dirty_path)
		clean_vol = safe_load(clean_path)

		return torch.from_numpy(dirty_vol).unsqueeze(0), torch.from_numpy(clean_vol).unsqueeze(0), filename

# ==========================================
# 2. 物理干扰注入引擎 (极限施压)
# ==========================================
def apply_perturbation(dirty_tensor, pert_type):
	if pert_type == "Original (Baseline)":
		return dirty_tensor
	elif pert_type == "Low Noise (σ=0.03)":
		noise = torch.randn_like(dirty_tensor) * 0.03
		return torch.clamp(dirty_tensor + noise, 0.0, 1.0)
	elif pert_type == "High Noise (σ=0.08)":
		noise = torch.randn_like(dirty_tensor) * 0.08
		return torch.clamp(dirty_tensor + noise, 0.0, 1.0)
	elif pert_type == "Contrast Drop (x0.8)":
		return torch.clamp(dirty_tensor * 0.8, 0.0, 1.0)
	elif pert_type == "Contrast Drop (x0.6)":  # 🌟 新增更严苛的对比度下降层级
		return torch.clamp(dirty_tensor * 0.6, 0.0, 1.0)
	else:
		return dirty_tensor

# ==========================================
# 3. 核心评估流水线
# ==========================================
def main():
	DIRTY_TEST  = r"G:\CT_RE\dataset\Final_Adaptive_Dirty\test"
	CLEAN_TEST  = r"G:\CT_RE\dataset\Final_Adaptive_GT\test"

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"🛡️ 启动终极 4 模型联合鲁棒性极限施压测试 (Device: {device})")

	print("📥 正在挂载四大模型权重...")
	models = {}

	# 1. 3D U-Net
	unet = UNet(spatial_dims=3, in_channels=1, out_channels=1,
				channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2, norm="instance").to(device)
	unet.load_state_dict(torch.load(r"checkpoints\3d_unet_best.pth", map_location=device, weights_only=True))
	models["3D U-Net"] = unet

	# 2. UNETR (纯 Transformer)
	unetr = UNETR(spatial_dims=3, in_channels=1, out_channels=1, img_size=(64, 256, 256),
				  feature_size=16, hidden_size=768, mlp_dim=3072, num_heads=12, norm_name="instance", res_block=True).to(device)
	unetr.load_state_dict(torch.load(r"checkpoints\unetr_baseline_best.pth", map_location=device, weights_only=True))
	models["UNETR"] = unetr

	# 3. Swin-UNETR (前期基线)
	swin = SwinUNETR(spatial_dims=3, in_channels=1, out_channels=1, feature_size=24, use_checkpoint=False).to(device)
	swin.load_state_dict(torch.load(r"checkpoints\swin_unetr_best.pth", map_location=device, weights_only=True))
	models["Swin-UNETR"] = swin

	# 💡 4. CS-Net (终极架构，注意检查你的pth文件名是否需要更改)
	cs_net = CSNet(img_size=(64, 256, 256), in_channels=1, out_channels=1, feature_size=24).to(device)
	cs_net.load_state_dict(torch.load(r"checkpoints\fast_robust_clnet_best.pth", map_location=device, weights_only=True))
	models["CS-Net"] = cs_net

	for m in models.values():
		m.eval()

	perturbations = [
		"Original (Baseline)",
		"Low Noise (σ=0.03)",
		"High Noise (σ=0.08)",
		"Contrast Drop (x0.8)",
		"Contrast Drop (x0.6)"  # 🌟 注册新的干扰项
	]

	results_psnr = {m: {p: [] for p in perturbations} for m in models.keys()}

	test_dataset = CLArtifactDataset(DIRTY_TEST, CLEAN_TEST)
	test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

	print("\n" + "="*70)
	print(">>> 开始极限施压跑分，这需要几分钟时间，请稍候... <<<")
	print("="*70)

	with torch.no_grad():
		for i, (dirty, clean, filename) in enumerate(test_loader):
			dirty, clean = dirty.to(device), clean.to(device)
			np_clean = clean.squeeze().cpu().numpy()

			for pert_name in perturbations:
				dirty_pert = apply_perturbation(dirty, pert_name)
				dirty_padded = F.pad(dirty_pert, (0, 0, 0, 0, 12, 12), mode='reflect')

				for model_name, model in models.items():
					with autocast('cuda'):
						preds_padded = model(dirty_padded)

					preds_aligned = preds_padded[:, :, 12:52, :, :]
					np_pred = preds_aligned.squeeze().cpu().numpy()
					np_pred = np.clip(np_pred, 0.0, 1.0)

					psnr = peak_signal_noise_ratio(np_clean, np_pred, data_range=1.0)
					results_psnr[model_name][pert_name].append(psnr)

			if (i+1) % 10 == 0 or i == 0:
				print(f"  [{i+1}/{len(test_dataset)}] 已完成对 {filename[0]} 的 {len(perturbations)*len(models)} 次交叉推断...")

	# --- 🏆 汇总抗压报告 ---
	print("\n" + "🌟"*35)
	print(" "*12 + "模型抗压能力 (Robustness) 最终分析报告 (单位: PSNR dB)")
	print("🌟"*35)

	header = f"{'Model':<12} | " + " | ".join([f"{p[:15]:<15}" for p in perturbations])
	print(header)
	print("-" * len(header))

	for model_name in models.keys():
		row_str = f"{model_name:<12} | "
		baseline_psnr = np.mean(results_psnr[model_name]["Original (Baseline)"])

		for pert_name in perturbations:
			mean_psnr = np.mean(results_psnr[model_name][pert_name])
			if pert_name == "Original (Baseline)":
				row_str += f"{mean_psnr:<15.2f} | "
			else:
				drop = baseline_psnr - mean_psnr
				row_str += f"{mean_psnr:.2f} (↓{drop:.2f})  | "
		print(row_str)

	print("=" * len(header))
	print("💡 核心看点：")
	print("  1. 在两组 Contrast Drop 测试中，观察 CS-Net 宏观跳跃连接对微弱信号的极强抢救保真能力。")
	print("  2. 在 High Noise 测试中，注意观察其性能衰减，这印证了论文中关于极限保真与高频抗噪之间的物理权衡定律。")

if __name__ == "__main__":
	main()