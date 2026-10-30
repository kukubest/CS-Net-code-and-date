import os
import glob
import numpy as np
import astra
import time

# ==========================================
# 1. CL 物理向量几何引擎 (保持与生成脏图绝对一致)
# ==========================================
def create_cl_vector_geometry(num_projections, tilt_angle_deg, SO, OD, det_spacing):
	alpha = np.radians(tilt_angle_deg)
	vectors = np.zeros((num_projections, 12))
	angles = np.linspace(0, 2 * np.pi, num_projections, endpoint=False)

	for i, theta in enumerate(angles):
		sx = SO * np.sin(alpha) * np.cos(theta)
		sy = SO * np.sin(alpha) * np.sin(theta)
		sz = SO * np.cos(alpha)

		dx = -OD * np.sin(alpha) * np.cos(theta)
		dy = -OD * np.sin(alpha) * np.sin(theta)
		dz = -OD * np.cos(alpha)

		ux = -np.sin(theta) * det_spacing
		uy = np.cos(theta) * det_spacing
		uz = 0

		ray_dir = np.array([dx - sx, dy - sy, dz - sz])
		ray_dir = ray_dir / np.linalg.norm(ray_dir)
		u_vec = np.array([ux, uy, uz])
		v_vec = np.cross(ray_dir, u_vec)
		v_vec = v_vec / np.linalg.norm(v_vec) * det_spacing

		vectors[i, 0:3] = [sx, sy, sz]
		vectors[i, 3:6] = [dx, dy, dz]
		vectors[i, 6:9] = u_vec
		vectors[i, 9:12] = v_vec

	return vectors

def add_poisson_noise(sinogram, I0=5e4):
	proj_photons = I0 * np.exp(-sinogram)
	noisy_photons = np.random.poisson(proj_photons)
	noisy_photons[noisy_photons == 0] = 1
	noisy_sino = -np.log(noisy_photons / I0)
	return noisy_sino

# ==========================================
# 2. 核心引擎：ASTRA FDK 解析重建
# ==========================================
def simulate_and_reconstruct_fdk(clean_vol, vol_geom, proj_geom):
	# --- 阶段 1：正向投影 ---
	sino_id, sino_data = astra.create_sino3d_gpu(clean_vol, proj_geom, vol_geom)

	# --- 阶段 2：物理噪声注入 ---
	noisy_sino_data = add_poisson_noise(sino_data, I0=5e4)
	noisy_sino_id = astra.data3d.create('-sino', proj_geom, noisy_sino_data)

	# --- 阶段 3：FDK 解析重建 ---
	# FDK 是解析算法，利用滤波反投影直接求解，速度极快，不需要迭代次数
	recon_id = astra.data3d.create('-vol', vol_geom)
	cfg = astra.astra_dict('FDK_CUDA')
	cfg['ReconstructionDataId'] = recon_id
	cfg['ProjectionDataId'] = noisy_sino_id

	alg_id = astra.algorithm.create(cfg)
	astra.algorithm.run(alg_id)

	# 提取体积数据
	fdk_vol = astra.data3d.get(recon_id)

	# 释放显存
	astra.algorithm.delete(alg_id)
	astra.data3d.delete(recon_id)
	astra.data3d.delete(sino_id)
	astra.data3d.delete(noisy_sino_id)

	return fdk_vol

# ==========================================
# 3. 批量生成与验证保存
# ==========================================
def batch_generate_fdk_baseline():
	# 路径配置
	CLEAN_TEST_DIR = r"G:\CT_RE\dataset\Final_Adaptive_GT\test"
	SAVE_DIR = r"G:\CT_RE\dataset\Final_Predictions"
	shape = (40, 256, 256)

	os.makedirs(SAVE_DIR, exist_ok=True)
	file_list = sorted(glob.glob(os.path.join(CLEAN_TEST_DIR, "*.raw")))

	if not file_list:
		print(f"❌ 未在 {CLEAN_TEST_DIR} 找到测试集数据！")
		return

	print(f"🚀 启动 FDK 解析重建基准测试，共 {len(file_list)} 个数据块...")

	# CL 系统几何参数 (严格匹配你之前的 45 度倾角配置)
	num_projections = 180
	tilt_angle = 45.0
	SO = 500.0
	OD = 500.0
	det_spacing = 1.0
	det_rows, det_cols = 300, 300

	vol_geom = astra.create_vol_geom(shape[2], shape[1], shape[0])
	vectors = create_cl_vector_geometry(num_projections, tilt_angle, SO, OD, det_spacing)
	proj_geom = astra.create_proj_geom('cone_vec', det_rows, det_cols, vectors)

	for i, clean_path in enumerate(file_list):
		start_time = time.time()
		filename = os.path.basename(clean_path)

		clean_vol = np.fromfile(clean_path, dtype=np.float32).reshape(shape)

		# 执行 FDK
		fdk_vol = simulate_and_reconstruct_fdk(clean_vol, vol_geom, proj_geom)

		# 💡 数据归一化对齐：强制拉回到 [0, 1]，以保证所有模型指标对比的公平性
		fdk_vol = (fdk_vol - np.min(fdk_vol)) / (np.max(fdk_vol) - np.min(fdk_vol) + 1e-8)

		# 保存结果为 _fdk_pred.raw
		save_name = filename.replace(".raw", "_fdk_pred.raw")
		save_path = os.path.join(SAVE_DIR, save_name)

		# 使用 float32 存档
		fdk_vol.astype(np.float32).tofile(save_path)

		elapsed = time.time() - start_time
		if (i + 1) % 10 == 0 or i == 0:
			print(f"  [{i+1}/{len(file_list)}] 已完成 FDK 重建: {save_name} (耗时: {elapsed:.2f}s)")

	print(f"\n🎉 传统物理基线 (FDK) 全部生成完毕！结果已保存在 {SAVE_DIR}")

if __name__ == "__main__":
	batch_generate_fdk_baseline()