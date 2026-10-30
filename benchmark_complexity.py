import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import time
import torch
import torch.nn as nn
import numpy as np
from thop import profile
from monai.networks.nets import UNet, UNETR, SwinUNETR

# ==========================================
# 🌟 插入我们自定义的终极架构 (FastRobustCLNet)
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

		# 核心引擎，使用 spatial_dims=3 适应下采样后的动态尺寸
		self.backbone = SwinUNETR(
			spatial_dims=3,
			in_channels=feature_size,
			out_channels=feature_size,
			feature_size=feature_size,
			use_checkpoint=False # 测速时务必关闭检查点以全速运行
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
# 📊 测速核心引擎
# ==========================================
def benchmark_model(model_name, model, device, input_shape=(1, 1, 64, 256, 256), num_runs=50):
	"""测试单个模型的各项硬件级性能指标"""
	model = model.to(device)
	model.eval()

	dummy_input = torch.randn(input_shape).to(device)

	# 1. 测参数量与 FLOPs
	macs, params = profile(model, inputs=(dummy_input,), verbose=False)
	params_m = params / 1e6
	flops_g = (macs * 2) / 1e9

	# 2. 测真实推理时间
	with torch.no_grad():
		for _ in range(10): # Warm-up
			_ = model(dummy_input)

	torch.cuda.synchronize()
	start_time = time.time()

	with torch.no_grad():
		for _ in range(num_runs):
			_ = model(dummy_input)

	torch.cuda.synchronize()
	end_time = time.time()

	avg_time_ms = ((end_time - start_time) / num_runs) * 1000

	del model
	del dummy_input
	torch.cuda.empty_cache()

	return params_m, flops_g, avg_time_ms

def main():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"🚀 启动硬件级模型基准测试 (Device: {device})")
	print("⏳ 正在预热显卡并计算复杂矩阵，请稍候...\n")

	# --- 1. 初始化所有对比模型 ---
	unet = UNet(
		spatial_dims=3, in_channels=1, out_channels=1,
		channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
		num_res_units=2, norm="instance"
	)

	unetr = UNETR(
		spatial_dims=3, in_channels=1, out_channels=1,
		img_size=(64, 256, 256), feature_size=16,
		hidden_size=768, mlp_dim=3072, num_heads=12,
		norm_name="instance", res_block=True
	)

	swin_unetr = SwinUNETR(
		spatial_dims=3, in_channels=1, out_channels=1,
		feature_size=24, use_checkpoint=False
	)

	# 💡 挂载终极模型
	fast_robust = FastRobustCLNet(
		img_size=(64, 256, 256),
		in_channels=1, out_channels=1, feature_size=24
	)

	# --- 2. 开始竞赛 ---
	models = {
		"3D U-Net (Baseline CNN)": unet,
		"UNETR (Pure ViT)": unetr,
		"Swin-UNETR (Early Baseline)": swin_unetr,
		"FastRobustCLNet (Ultimate)": fast_robust  # 💡 终极架构登场
	}

	results = {}
	for name, model in models.items():
		print(f"[{name}] 正在压榨算力测试中...")
		p, f, t = benchmark_model(name, model, device)
		results[name] = {"Params (M)": p, "FLOPs (G)": f, "Time (ms)": t}

	# --- 3. 打印最终学术风数据表格 ---
	print("\n" + "="*75)
	print(f"{'Model Architecture':<30} | {'Params (M) ↓':<12} | {'FLOPs (G) ↓':<12} | {'Time (ms/vol) ↓'}")
	print("-" * 75)

	for name, metrics in results.items():
		p = metrics["Params (M)"]
		f = metrics["FLOPs (G)"]
		t = metrics["Time (ms)"]
		print(f"{name:<30} | {p:>10.2f} | {f:>10.2f} | {t:>13.2f}")

	print("="*75)
	print("💡 论文解读指南：")
	print("  观察最后一名选手 (FastRobustCLNet) 的 FLOPs 和 Time！")
	print("  由于前置了 8 倍的体积下采样，它的计算量将出现【断崖式下降】，这正是其实时推理的底气。")

if __name__ == "__main__":
	main()