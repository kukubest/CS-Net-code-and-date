import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

class FastRobustCLNet(nn.Module):
	"""
	极速抗噪混合架构 (Fast & Robust CNN-Swin Architecture)
	通过【下采样前置】极大降低 FLOPs，并通过【宏观跳跃连接】保护极高频工业边缘
	"""
	def __init__(self, img_size=(64, 256, 256), in_channels=1, out_channels=1, feature_size=24):
		super().__init__()

		# ==========================================
		# 💡 策略一：下采样前置 (Downsampling Stem)
		# ==========================================
		# 1. 高频特征提取与抗噪网关 (保持原分辨率 64x256x256)
		self.stem_enc = nn.Sequential(
			nn.Conv3d(in_channels, feature_size, kernel_size=3, stride=1, padding=1),
			nn.InstanceNorm3d(feature_size),
			nn.LeakyReLU(inplace=True)
		)

		# 2. 空间降维层 (长宽高各缩小 2 倍 -> 整体体积缩小 8 倍！)
		# 输出尺寸变为: 32x128x128
		self.downsample = nn.Sequential(
			nn.Conv3d(feature_size, feature_size, kernel_size=3, stride=2, padding=1),
			nn.InstanceNorm3d(feature_size),
			nn.LeakyReLU(inplace=True)
		)

		# 计算降维后给 Swin-UNETR 的真实尺寸
		downsampled_size = (img_size[0] // 2, img_size[1] // 2, img_size[2] // 2)

		# ==========================================
		# 🧠 核心推理引擎：Swin-UNETR (工作在低分辨率隐空间)
		# 这里的 FLOPs 和显存开销直接骤降了将近 8 倍！
		# ==========================================
		self.backbone = SwinUNETR(
			img_size=downsampled_size,
			in_channels=feature_size,
			out_channels=feature_size,  # 输出特征数保持一致
			feature_size=feature_size,
			use_checkpoint=False
		)

		# ==========================================
		# 🎯 物理分辨率还原与融合
		# ==========================================
		# 3. 空间升维层 (32x128x128 还原回 64x256x256)
		self.upsample = nn.Sequential(
			nn.ConvTranspose3d(feature_size, feature_size, kernel_size=2, stride=2),
			nn.InstanceNorm3d(feature_size),
			nn.LeakyReLU(inplace=True)
		)

		# 4. 宏观解码器 (接收跳跃连接，还原锐利边缘)
		# 输入通道数是 feature_size * 2 (因为拼接了第一步的高频特征)
		self.tail_dec = nn.Sequential(
			nn.Conv3d(feature_size * 2, feature_size, kernel_size=3, stride=1, padding=1),
			nn.InstanceNorm3d(feature_size),
			nn.LeakyReLU(inplace=True),
			# 最终映射回单通道密度图
			nn.Conv3d(feature_size, out_channels, kernel_size=3, stride=1, padding=1)
		)

	def forward(self, x):
		# 阶段 A: 获取干净的高频边缘特征 (备份，留给最后融合)
		x_high_res = self.stem_enc(x)

		# 阶段 B: 空间降维 (体积缩减 87.5%)
		x_down = self.downsample(x_high_res)

		# 阶段 C: Swin 引擎进行全局伪影与拖尾的拔除 (速度极快)
		swin_out = self.backbone(x_down)

		# 阶段 D: 升维回到物理空间
		x_up = self.upsample(swin_out)

		# 阶段 E: 宏观跳跃连接拼接 (Concat) + 最终解码
		# 将低频拓扑(x_up)与高频边缘(x_high_res)完美融合
		fusion = torch.cat([x_up, x_high_res], dim=1)
		out = self.tail_dec(fusion)

		return out