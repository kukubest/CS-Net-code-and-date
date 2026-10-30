import os
import glob
import numpy as np
import astra
import random
import time

def create_cl_vector_geometry(num_projections, tilt_angle_deg, SO, OD, det_spacing):
	"""
	建立符合物理实际的旋转计算层析 (CL) 向量几何
	SO: 射线源到旋转中心的距离
	OD: 旋转中心到探测器的距离
	tilt_angle_deg: CL倾角 (如 60度)
	"""
	alpha = np.radians(tilt_angle_deg)
	vectors = np.zeros((num_projections, 12))
	angles = np.linspace(0, 2 * np.pi, num_projections, endpoint=False)

	for i, theta in enumerate(angles):
		# 1. 射线源的空间坐标 (Source)
		sx = SO * np.sin(alpha) * np.cos(theta)
		sy = SO * np.sin(alpha) * np.sin(theta)
		sz = SO * np.cos(alpha)

		# 2. 探测器中心的空间坐标 (Detector)
		dx = -OD * np.sin(alpha) * np.cos(theta)
		dy = -OD * np.sin(alpha) * np.sin(theta)
		dz = -OD * np.cos(alpha)

		# 3. 探测器的水平朝向向量 (U 向量)
		ux = -np.sin(theta) * det_spacing
		uy = np.cos(theta) * det_spacing
		uz = 0

		# 4. 探测器的垂直朝向向量 (V 向量) - 垂直于 U 和 射线方向
		ray_dir = np.array([dx - sx, dy - sy, dz - sz])
		ray_dir = ray_dir / np.linalg.norm(ray_dir)
		u_vec = np.array([ux, uy, uz])
		v_vec = np.cross(ray_dir, u_vec)
		v_vec = v_vec / np.linalg.norm(v_vec) * det_spacing

		# 填入 ASTRA 向量矩阵
		vectors[i, 0:3] = [sx, sy, sz]
		vectors[i, 3:6] = [dx, dy, dz]
		vectors[i, 6:9] = u_vec
		vectors[i, 9:12] = v_vec

	return vectors

def add_poisson_noise(sinogram, I0=5e4):
	"""
	注入泊松量子噪声，模拟真实的 X 射线光子衰减波动
	I0 越小，噪声越大。5e4 是一个中等强度的典型工业设定
	"""
	# 模拟射线穿过物体后的光子数
	proj_photons = I0 * np.exp(-sinogram)
	# 添加泊松噪声
	noisy_photons = np.random.poisson(proj_photons)
	noisy_photons[noisy_photons == 0] = 1  # 防止取对数时报错
	# 还原为衰减弦图
	noisy_sino = -np.log(noisy_photons / I0)
	return noisy_sino

def simulate_and_reconstruct(clean_vol, vol_geom, proj_geom, num_iterations=30):
	"""ASTRA 物理正向投影 -> 加噪 -> SART/SIRT 重建"""

	# --- 阶段 1：正向投影 (生成弦图) ---
	# 创建投影数据对象
	sino_id, sino_data = astra.create_sino3d_gpu(clean_vol, proj_geom, vol_geom)

	# --- 阶段 2：注入真实物理噪声 ---
	noisy_sino_data = add_poisson_noise(sino_data, I0=5e4)

	# 将带噪声的数据重新绑回 ASTRA 内存
	noisy_sino_id = astra.data3d.create('-sino', proj_geom, noisy_sino_data)

	# --- 阶段 3：迭代重建 (产生缺失锥伪影) ---
	# 初始化一个空的 3D 重建结果矩阵
	recon_id = astra.data3d.create('-vol', vol_geom)

	# 使用 SIRT 或 SART 进行有限迭代重建 (迭代次数少会加剧混叠伪影，更符合真实快扫场景)
	cfg = astra.astra_dict('SIRT3D_CUDA')
	cfg['ReconstructionDataId'] = recon_id
	cfg['ProjectionDataId'] = noisy_sino_id

	alg_id = astra.algorithm.create(cfg)
	astra.algorithm.run(alg_id, num_iterations)

	# 获取脏数据矩阵
	dirty_vol = astra.data3d.get(recon_id)

	# 释放 ASTRA 显存！极其重要，否则会显存溢出
	astra.algorithm.delete(alg_id)
	astra.data3d.delete(recon_id)
	astra.data3d.delete(sino_id)
	astra.data3d.delete(noisy_sino_id)

	return dirty_vol

def batch_generate_dirty_dataset(clean_dir, dirty_dir, shape=(40, 256, 256)):
	"""批量处理文件夹内的所有 clean raw 文件"""
	os.makedirs(dirty_dir, exist_ok=True)

	file_list = sorted(glob.glob(os.path.join(clean_dir, "*.raw")))
	if not file_list:
		print(f"❌ 未在 {clean_dir} 找到数据集！")
		return

	print(f"🚀 准备开始 CL 物理仿真，共 {len(file_list)} 个数据块...")

	# ==========================================
	# 📐 设定 CL 系统的几何参数
	# 参考你论文中的典型参数配置
	# ==========================================
	num_projections = 180       # 投影角度数量
	tilt_angle = 45.0           # CL 倾角 (度)
	SO = 500.0                  # 源到中心距离
	OD = 500.0                  # 中心到探测器距离
	det_spacing = 1.0           # 探测器像素大小

	# 探测器长宽 (为了包裹住整个样本，探测器稍微设大一点)
	det_rows = 300
	det_cols = 300

	# 创建 ASTRA 体积几何和投影几何
	vol_geom = astra.create_vol_geom(shape[2], shape[1], shape[0]) # ASTRA 顺序是 x, y, z
	vectors = create_cl_vector_geometry(num_projections, tilt_angle, SO, OD, det_spacing)
	proj_geom = astra.create_proj_geom('cone_vec', det_rows, det_cols, vectors)

	for i, clean_path in enumerate(file_list):
		start_time = time.time()
		filename = os.path.basename(clean_path)

		# 1. 读取干净数据
		clean_vol = np.fromfile(clean_path, dtype=np.float32).reshape(shape)

		# 2. 核心仿真引擎：生成脏数据
		# 迭代次数设为 30，迭代越少由于不完全数据导致的慧星拖尾越严重
		dirty_vol = simulate_and_reconstruct(clean_vol, vol_geom, proj_geom, num_iterations=30)

		# 3. 数据归一化 (将脏数据的数值范围强制拉回到和干净数据同频的 [0, 1])
		# 这一步对深度学习极其重要！
		dirty_vol = (dirty_vol - np.min(dirty_vol)) / (np.max(dirty_vol) - np.min(dirty_vol) + 1e-8)

		# 4. 存档
		save_path = os.path.join(dirty_dir, filename.replace(".raw", "_dirty.raw"))
		dirty_vol.astype(np.float32).tofile(save_path)

		elapsed = time.time() - start_time
		if (i + 1) % 10 == 0 or i == 0:
			print(f"  [{i+1}/{len(file_list)}] 已生成脏数据: {filename} (耗时: {elapsed:.2f}秒/个)")

	print(f"\n🎉 恭喜！全套物理级脏数据已生成完毕！保存在: {dirty_dir}")

if __name__ == "__main__":
	# ---------------- 训练集生成 ----------------
	CLEAN_TRAIN_DIR = r"G:\CT_RE\dataset\Final_Adaptive_GT\train"
	DIRTY_TRAIN_DIR = r"G:\CT_RE\dataset\Final_Adaptive_Dirty\train"
	batch_generate_dirty_dataset(CLEAN_TRAIN_DIR, DIRTY_TRAIN_DIR)

	# ---------------- 测试集生成 ----------------
	CLEAN_TEST_DIR = r"G:\CT_RE\dataset\Final_Adaptive_GT\test"
	DIRTY_TEST_DIR = r"G:\CT_RE\dataset\Final_Adaptive_Dirty\test"
	batch_generate_dirty_dataset(CLEAN_TEST_DIR, DIRTY_TEST_DIR)