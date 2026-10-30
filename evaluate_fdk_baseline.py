import os
import glob
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error

def main():
	# --- ⚙️ 路径配置 ---
	CLEAN_TEST_DIR = r"G:\CT_RE\dataset\Final_Adaptive_GT\test"
	# FDK 预测结果的保存路径
	PRED_DIR = r"G:\CT_RE\dataset\Final_Predictions"
	shape = (40, 256, 256)

	clean_files = sorted(glob.glob(os.path.join(CLEAN_TEST_DIR, "*.raw")))

	if not clean_files:
		print(f"❌ 未找到测试集真实标签，请检查路径！")
		return

	print("🚀 启动传统物理基线 (FDK) 评估引擎...")

	# --- 📊 性能记录指标 ---
	metrics = {'fdk_rmse': [], 'fdk_psnr': [], 'fdk_ssim': []}

	print("\n" + "="*50)
	print(">>> 开始进行 FDK 解析重建跑分测试 <<<")
	print("="*50)

	for i, clean_path in enumerate(clean_files):
		filename = os.path.basename(clean_path)

		# 寻找对应的 FDK 结果文件
		fdk_name = filename.replace(".raw", "_fdk_pred.raw")
		fdk_path = os.path.join(PRED_DIR, fdk_name)

		if not os.path.exists(fdk_path):
			print(f"⚠️ 找不到对应的 FDK 结果文件: {fdk_name}，已跳过。")
			continue

		# 读取 32 位浮点数矩阵
		clean_vol = np.fromfile(clean_path, dtype=np.float32).reshape(shape)
		fdk_vol = np.fromfile(fdk_path, dtype=np.float32).reshape(shape)

		# 确保数据在 [0, 1] 范围内，防止极个别噪点影响指标
		fdk_vol = np.clip(fdk_vol, 0.0, 1.0)

		# --- 📈 计算量化指标 ---
		rmse = np.sqrt(mean_squared_error(clean_vol, fdk_vol))
		psnr = peak_signal_noise_ratio(clean_vol, fdk_vol, data_range=1.0)
		ssim = structural_similarity(clean_vol, fdk_vol, data_range=1.0)

		metrics['fdk_rmse'].append(rmse)
		metrics['fdk_psnr'].append(psnr)
		metrics['fdk_ssim'].append(ssim)

		if (i+1) % 20 == 0 or i == 0:
			print(f"  [{i+1}/{len(clean_files)}] {filename} | "
				  f"FDK PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}")

	# --- 🏆 汇总报告 ---
	print("\n" + "🌟"*25)
	print(" "*8 + "传统物理基线 (FDK) 最终平均性能报告")
	print("🌟"*25)

	print(f"【RMSE (均方根误差)】")
	print(f"  > FDK 解析重建: {np.mean(metrics['fdk_rmse']):.4f}")

	print(f"\n【PSNR (峰值信噪比)】")
	print(f"  > FDK 解析重建: {np.mean(metrics['fdk_psnr']):.2f} dB")

	print(f"\n【SSIM (结构相似度)】")
	print(f"  > FDK 解析重建: {np.mean(metrics['fdk_ssim']):.4f}")
	print("🌟"*25)

if __name__ == "__main__":
	main()