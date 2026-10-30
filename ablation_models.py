"""Canonical model definitions for the five CS-Net ablation groups."""

import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR


class CSNet(nn.Module):
	"""Project CS-Net with explicit compression/HR-skip ablation switches."""

	def __init__(self, in_channels=1, out_channels=1, feature_size=24,
				 use_checkpoint=True, compression=True, hr_skip=True):
		super().__init__()
		self.compression = bool(compression)
		self.hr_skip = bool(hr_skip)
		self.stem_enc = nn.Sequential(
			nn.Conv3d(in_channels, feature_size, kernel_size=3, stride=1, padding=1),
			nn.InstanceNorm3d(feature_size),
			nn.LeakyReLU(inplace=True),
		)
		self.downsample = nn.Sequential(
			nn.Conv3d(feature_size, feature_size, kernel_size=3, stride=2, padding=1),
			nn.InstanceNorm3d(feature_size),
			nn.LeakyReLU(inplace=True),
		)
		self.backbone = SwinUNETR(
			spatial_dims=3,
			in_channels=feature_size,
			out_channels=feature_size,
			feature_size=feature_size,
			use_checkpoint=use_checkpoint,
		)
		self.upsample = nn.Sequential(
			nn.ConvTranspose3d(feature_size, feature_size, kernel_size=2, stride=2),
			nn.InstanceNorm3d(feature_size),
			nn.LeakyReLU(inplace=True),
		)
		tail_channels = feature_size * 2 if self.hr_skip else feature_size
		self.tail_dec = nn.Sequential(
			nn.Conv3d(tail_channels, feature_size, kernel_size=3, stride=1, padding=1),
			nn.InstanceNorm3d(feature_size),
			nn.LeakyReLU(inplace=True),
			nn.Conv3d(feature_size, out_channels, kernel_size=3, stride=1, padding=1),
		)

	def forward(self, x):
		high_res = self.stem_enc(x)
		if self.compression:
			latent = self.downsample(high_res)
			latent = self.backbone(latent)
			decoded = self.upsample(latent)
		else:
			# Registered compression modules are intentionally bypassed.  This
			# preserves D's parameter set while isolating spatial compression.
			decoded = self.backbone(high_res)
		if self.hr_skip:
			decoded = torch.cat((decoded, high_res), dim=1)
		return self.tail_dec(decoded)


def build_model(experiment, model_config, checkpoint_override=None):
	"""Build exactly the architecture named by a centralized experiment config."""
	arch = experiment["architecture"]
	use_checkpoint = model_config["use_checkpoint"]
	if checkpoint_override is not None:
		use_checkpoint = bool(checkpoint_override)
	common = {
		"in_channels": model_config["in_channels"],
		"out_channels": model_config["out_channels"],
		"feature_size": model_config["feature_size"],
		"use_checkpoint": use_checkpoint,
	}
	if arch == "swin_unetr_baseline":
		return SwinUNETR(spatial_dims=3, **common)
	if arch == "csnet":
		return CSNet(compression=True, hr_skip=True, **common)
	if arch == "csnet_no_compression":
		return CSNet(compression=False, hr_skip=True, **common)
	if arch == "csnet_compressed_no_hr_skip":
		return CSNet(compression=True, hr_skip=False, **common)
	raise ValueError("Unknown architecture: {}".format(arch))


def build_loss(name):
	if name.lower() == "l1":
		return nn.L1Loss()
	if name.lower() == "mse":
		return nn.MSELoss()
	raise ValueError("Unknown loss: {}".format(name))
