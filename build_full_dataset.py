import numpy as np
from PIL import Image
import os
import random

def load_grayscale_native(image_path):
	"""【自适应修改】直接读取原图，保留最原始的物理分辨率和长宽比"""
	if not os.path.exists(image_path):
		return None

	img = Image.open(image_path).convert('L')
	return np.array(img)

def process_and_resize_patch(patch_array, target_size=256, threshold=100):
	"""将任意尺寸的灰度裁剪块，平滑缩放到 256x256 并二值化"""
	if patch_array.shape[0] == target_size:
		return patch_array > threshold

	img = Image.fromarray(patch_array)
	img_resized = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
	return np.array(img_resized) > threshold

def build_3d_patch_4layers(top_mask, bot_mask, drill_mask, solder_mask, shape=(40, 256, 256)):
	"""将 4 层 2D 掩膜压制成 40 层的 3D 物理体素"""
	Z, Y, X = shape
	vol = np.zeros(shape, dtype=np.float32)

	# 1. 基板
	vol[10:30, :, :] = 0.21
	# 2. 铜层
	for z in range(30, 32): vol[z, top_mask] = 1.00
	for z in range(8, 10):  vol[z, bot_mask] = 1.00
	# 3. 锡膏
	for z in range(32, 35): vol[z, solder_mask] = 0.82
	# 4. 打孔
	for z in range(0, 40):  vol[z, drill_mask] = 0.00

	return vol

def process_single_board(board_idx, base_pcb_dir, save_dir, num_patches=200, target_size=256):
	"""处理单个 PCB 文件夹，自适应尺寸进行裁剪"""
	os.makedirs(save_dir, exist_ok=True)

	board_dir = os.path.join(base_pcb_dir, str(board_idx))
	print(f"\n📁 正在处理第 {board_idx} 组 PCB 数据 ({board_dir})...")

	top_file = os.path.join(board_dir, f"t{board_idx}.jpg")
	bot_file = os.path.join(board_dir, f"b{board_idx}.jpg")
	drill_file = os.path.join(board_dir, f"d{board_idx}.jpg")
	solder_file = os.path.join(board_dir, f"s{board_idx}.jpg")

	# 【自适应修改 1】以顶层铜作为基准，获取当前板子的真实分辨率
	top_gray = load_grayscale_native(top_file)
	if top_gray is None:
		print(f"  [❌错误] 未找到顶层线路图 {top_file}，跳过此板。")
		return

	canvas_shape = top_gray.shape # 获得 (Height, Width)
	print(f"  -> 检测到真实物理分辨率: {canvas_shape[1]} x {canvas_shape[0]} (宽x高)")

	# 加载其他层。如果某一层缺失，则用全黑(无)的空矩阵代替，保持形状一致
	bot_gray = load_grayscale_native(bot_file)
	if bot_gray is None: bot_gray = np.zeros(canvas_shape, dtype=np.uint8)

	drill_gray = load_grayscale_native(drill_file)
	if drill_gray is None: drill_gray = np.zeros(canvas_shape, dtype=np.uint8)

	solder_gray = load_grayscale_native(solder_file)
	if solder_gray is None: solder_gray = np.zeros(canvas_shape, dtype=np.uint8)

	# 【自适应修改 2】动态计算当前板卡支持的金字塔尺度
	# 比如板子短边只有 800，那么 4x (1024) 的截取框就放不下，只能用 1x 和 2x
	max_crop_allowed = min(canvas_shape[0], canvas_shape[1])
	desired_scales = [1, 2, 4]
	valid_scales = [s for s in desired_scales if s * target_size <= max_crop_allowed]

	if not valid_scales:
		print(f"  [⚠️警告] 此板卡太小 (短边 {max_crop_allowed})，无法切出哪怕 {target_size} 的图，跳过。")
		return

	print(f"  -> 当前板卡支持的金字塔倍率: {[str(s)+'x' for s in valid_scales]}")

	valid_count = 0
	attempts = 0

	while valid_count < num_patches:
		attempts += 1
		if attempts > num_patches * 20:
			print(f"  [⚠️] 图纸走线过少或有效区域不够，已强制停止获取 (产出 {valid_count} 个)。")
			break

		# 在支持的尺度池里随机抽选
		current_scale = random.choice(valid_scales)
		crop_size = target_size * current_scale

		# 在真实的长宽范围内随机定位坐标
		start_y = random.randint(0, canvas_shape[0] - crop_size)
		start_x = random.randint(0, canvas_shape[1] - crop_size)

		# 绝对对齐抠图
		top_crop = top_gray[start_y:start_y+crop_size, start_x:start_x+crop_size]
		bot_crop = bot_gray[start_y:start_y+crop_size, start_x:start_x+crop_size]
		drill_crop = drill_gray[start_y:start_y+crop_size, start_x:start_x+crop_size]
		solder_crop = solder_gray[start_y:start_y+crop_size, start_x:start_x+crop_size]

		# 压缩并二值化
		top_mask = process_and_resize_patch(top_crop, target_size)
		bot_mask = process_and_resize_patch(bot_crop, target_size)
		drill_mask = process_and_resize_patch(drill_crop, target_size)
		solder_mask = process_and_resize_patch(solder_crop, target_size)

		# 智能细节过滤
		top_ratio, bot_ratio = np.mean(top_mask), np.mean(bot_mask)
		top_var, bot_var = np.var(top_mask), np.var(bot_mask)

		if top_ratio < 0.05 and bot_ratio < 0.05: continue
		if top_ratio > 0.95 and bot_ratio > 0.95: continue
		if top_var < 0.05 and bot_var < 0.05: continue

		# 压制 3D
		patch_3d = build_3d_patch_4layers(top_mask, bot_mask, drill_mask, solder_mask, shape=(40, target_size, target_size))

		filename = os.path.join(save_dir, f"board{board_idx}_patch{valid_count:04d}_scale{current_scale}x.raw")
		patch_3d.tofile(filename)
		valid_count += 1

		if valid_count % 50 == 0:
			print(f"  已精选 {valid_count}/{num_patches} 块数据 (累计尝试 {attempts} 次)")

def main():
	BASE_PCB_DIR = r"G:\CT_RE\PCB"
	DATASET_OUT_DIR = r"G:\CT_RE\dataset\Final_Adaptive_GT"
	TRAIN_DIR = os.path.join(DATASET_OUT_DIR, "train")
	TEST_DIR = os.path.join(DATASET_OUT_DIR, "test")

	PATCHES_PER_BOARD_TRAIN = 250
	PATCHES_PER_BOARD_TEST = 200

	print("🚀 开始构建多目标自适应数字孪生数据集...")

	# 1. 训练集 (1~4 组)
	print("\n" + "="*40)
	print(">>> 阶段一：生成训练集 (Train Set) <<<")
	print("="*40)
	for i in range(1, 5):
		process_single_board(i, BASE_PCB_DIR, TRAIN_DIR, PATCHES_PER_BOARD_TRAIN)

	# 2. 测试集 (第 5 组)
	print("\n" + "="*40)
	print(">>> 阶段二：生成测试集 (Test Set) <<<")
	print("="*40)
	process_single_board(5, BASE_PCB_DIR, TEST_DIR, PATCHES_PER_BOARD_TEST)

	print(f"\n🎉 自适应数据集成功构建在: {DATASET_OUT_DIR}")

if __name__ == "__main__":
	main()