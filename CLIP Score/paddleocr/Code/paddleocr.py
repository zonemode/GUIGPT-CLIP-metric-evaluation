import os
import sys
import json
import cv2
import numpy as np
from paddleocr import PaddleOCR
from typing import List, Dict, Any, Union

# === CUDA环境设置（必须放在最开头）===
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
cuda_bin_path = cuda_path + r"\bin"

# 设置环境变量
os.environ['CUDA_HOME'] = cuda_path
os.environ['PATH'] = cuda_bin_path + ';' + os.environ['PATH']

# Python 3.8+ 添加DLL目录
if hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(cuda_bin_path)

print("="*60)
print("CUDA环境设置完成")
print("="*60)
print(f"CUDA路径: {cuda_path}")
print(f"DLL目录: {cuda_bin_path}")

# 现在导入PaddlePaddle
try:
    import paddle
    print("✅ PaddlePaddle导入成功")
    print(f"版本: {paddle.__version__}")
    print(f"CUDA可用: {paddle.is_compiled_with_cuda()}")
    
    if paddle.is_compiled_with_cuda():
        print(f"GPU设备: {paddle.device.cuda.get_device_name()}")
        
except ImportError as e:
    print(f"❌❌ PaddlePaddle导入失败: {e}")
    sys.exit(1)

class PaddleOCRProcessor:
    def __init__(self, use_gpu=True):
        """
        初始化PaddleOCR处理器
        """
        self.use_gpu = use_gpu
        try:
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                use_gpu=use_gpu,
                lang='en',
                det_db_thresh=0.3,
                det_db_box_thresh=0.6,
                rec_score_thresh=0.5
            )
            print(f"PaddleOCR初始化成功，使用{'GPU' if use_gpu else 'CPU'}模式")
        except Exception as e:
            print(f"GPU模式初始化失败: {str(e)}")
            print("尝试使用CPU模式...")
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                use_gpu=False,  # 强制使用CPU
                lang='en',
                det_db_thresh=0.3,
                det_db_box_thresh=0.6,
                rec_score_thresh=0.5
            )
            print("PaddleOCR CPU模式初始化成功")
            self.use_gpu = False
    
    def _process_single_image(self, image_path: str) -> Dict[str, Any]:
        """
        处理单张图片，返回OCR识别结果（格式与文档1相同）
        注意：这是内部方法，不对外暴露
        """
        try:
            # 读取图片
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"无法读取图片: {image_path}")
    
            # 获取图片形状（与文档1格式一致）
            height, width, channels = img.shape
    
            # 执行OCR识别
            result = self.ocr.ocr(img, cls=True)
    
            # 处理识别结果
            ocr_data = self._format_ocr_result(result, width, height, channels)
    
            return ocr_data
    
        except Exception as e:
            print(f"处理图片 {image_path} 时出错: {str(e)}")
            return {
                "img_shape": [0, 0, 3],
                "texts": [],
                "error": str(e)
            }

    def _format_ocr_result(self, result: List, width: int, height: int, channels: int) -> Dict[str, Any]:
        """格式化OCR识别结果为与文档1相同的JSON格式"""
        # 修正：直接使用传递的参数
        img_shape_formatted = [height, width, channels]  # 格式: [宽度, 高度, 通道数]

        # 处理文本区域
        texts = []

        if result and result[0]:
            text_id = 0
            for line in result[0]:
                if line and len(line) >= 2:
                    # 提取坐标点
                    points = line[0]
                    text_content, confidence = line[1]
            
                    # 过滤低置信度的结果
                    if confidence < 0.5:
                        continue
            
                    # 过滤明显错误的文本（如单个字符、无意义字符等）
                    if self._should_filter_text(text_content):
                        continue
            
                    # 计算边界框坐标（与文档1格式一致）
                    points_array = np.array(points, dtype=np.int32)
                    x_coords = points_array[:, 0]
                    y_coords = points_array[:, 1]
            
                    # 确保坐标在合理范围内
                    column_min = max(0, int(np.min(x_coords)))
                    row_min = max(0, int(np.min(y_coords)))
                    column_max = min(width, int(np.max(x_coords)))
                    row_max = min(height, int(np.max(y_coords)))
            
                    # 计算宽度和高度（确保为正数）
                    width_box = max(1, column_max - column_min)
                    height_box = max(1, row_max - row_min)
            
                    # 过滤过小的文本区域（可能是噪声）
                    if width_box < 10 or height_box < 5:
                        continue
            
                    # 创建文本对象（与文档1格式完全一致）
                    text_obj = {
                        "id": text_id,
                        "content": text_content,
                        "column_min": column_min,
                        "row_min": row_min,
                        "column_max": column_max,
                        "row_max": row_max,
                        "width": width_box,
                        "height": height_box
                    }
            
                    texts.append(text_obj)
                    text_id += 1

        # 按位置排序文本，使其顺序更合理（先按行，再按列）
        texts.sort(key=lambda x: (x['row_min'], x['column_min']))

        # 重新分配ID以确保连续
        for idx, text in enumerate(texts):
            text['id'] = idx

        # 构建完整的OCR数据（与文档1格式完全一致）
        ocr_data = {
            "img_shape": img_shape_formatted,  # 这里已经是正确的格式
            "texts": texts
        }

        return ocr_data
    
    def _should_filter_text(self, text_content: str) -> bool:
        """
        判断是否应该过滤掉该文本
        """
        # 过滤空文本
        if not text_content or text_content.strip() == "":
            return True
        
        # 过滤纯数字（除非是特定格式如时间）
        if text_content.isdigit() and len(text_content) < 3:
            return True
        
        # 过滤连续的点或其他无意义字符
        meaningless_patterns = ['.........', '..', '...', '....', '.....', '......']
        if text_content in meaningless_patterns:
            return True
        
        # 过滤过短的文本（除非是常见单词）
        short_words = {'I', 'a', 'A', 'OK', 'ok', 'Hi', 'hi', 'No', 'no', 'Yes', 'yes'}
        if len(text_content) <= 1 and text_content not in short_words:
            return True
        
        # 过滤特殊字符
        if len(text_content.strip()) == 1 and not text_content.isalnum():
            return True
        
        return False
    
    def process_multiple_folders(self, folder_paths: Union[str, List[str]], output_base_dir: str = "ocr_results"):
        """
        处理多个文件夹中的所有图片
        """
        # 确保folder_paths是列表形式
        if isinstance(folder_paths, str):
            folder_paths = [folder_paths]
        
        # 创建输出基础目录
        os.makedirs(output_base_dir, exist_ok=True)
        
        # 处理每个文件夹
        all_results = {}
        for folder_path in folder_paths:
            if not os.path.exists(folder_path):
                print(f"文件夹不存在: {folder_path}")
                continue
                
            # 创建文件夹特定的输出目录
            folder_name = os.path.basename(folder_path.rstrip('/\\'))
            output_dir = os.path.join(output_base_dir, folder_name)
            
            print(f"\n处理文件夹: {folder_path}")
            print(f"输出目录: {output_dir}")
            
            # 处理当前文件夹
            folder_results = self._process_single_folder(folder_path, output_dir)
            all_results[folder_path] = folder_results
        
        # 保存所有结果的汇总
        summary_path = os.path.join(output_base_dir, "all_folders_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)
        
        print(f"\n所有文件夹处理完成！结果保存在: {output_base_dir}")
        return all_results
    
    def _process_single_folder(self, folder_path: str, output_dir: str) -> Dict[str, Any]:
        """
        处理单个文件夹中的所有图片
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 支持的图片格式
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # 遍历文件夹中的图片
        image_files = []
        for filename in os.listdir(folder_path):
            if any(filename.lower().endswith(ext) for ext in valid_extensions):
                image_files.append(os.path.join(folder_path, filename))
        
        print(f"找到 {len(image_files)} 张图片需要处理")
        
        # 处理每张图片
        folder_results = {}
        for i, image_path in enumerate(image_files, 1):
            print(f"  处理图片 {i}/{len(image_files)}: {os.path.basename(image_path)}")
            
            # 处理单张图片
            result = self._process_single_image(image_path)
            
            # 保存单个图片的结果
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}_ocr_result.json")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            
            folder_results[image_path] = result
            
            # 生成可视化结果
            self._generate_visualization(image_path, result, output_dir)
        
        # 保存当前文件夹的结果汇总
        summary_path = os.path.join(output_dir, "folder_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(folder_results, f, indent=4, ensure_ascii=False)
        
        return folder_results
    
    def _generate_visualization(self, image_path: str, ocr_data: Dict, output_dir: str):
        """
        生成带识别框的可视化图片
        优化：与图二一样，使用红色矩形框，不添加文本标签
        """
        try:
            # 读取原图
            img = cv2.imread(image_path)
            if img is None:
                print(f"无法读取图片: {image_path}")
                return
                
            # 创建原图的副本，不在原图上直接修改
            img_rgb = img.copy()
            
            # 绘制识别框 - 优化为红色矩形框，与图二一致
            for text_region in ocr_data.get('texts', []):
                # 提取边界框坐标
                x1 = text_region['column_min']
                y1 = text_region['row_min']
                x2 = text_region['column_max']
                y2 = text_region['row_max']
                
                # 绘制矩形框 - 使用红色 (BGR: 0, 0, 255)，线宽为2
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # 注意：图二中只有红色矩形框，没有文本标签
                # 因此不添加文本标签
            
            # 保存可视化结果
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}_visualization.png")
            
            # 保存图片
            cv2.imwrite(output_path, img_rgb)
            
        except Exception as e:
            print(f"生成可视化结果时出错: {str(e)}")

def process_multiple_folders_directly(folder_paths: Union[str, List[str]]):
    """
    直接处理多个文件夹
    """
    # 初始化处理器（先尝试GPU，失败后自动切换到CPU）
    processor = PaddleOCRProcessor(use_gpu=True)
    
    # 确保folder_paths是列表形式
    if isinstance(folder_paths, str):
        folder_paths = [folder_paths]
    
    # 检查文件夹是否存在
    valid_folders = []
    for folder_path in folder_paths:
        if not os.path.exists(folder_path):
            print(f"错误：文件夹路径不存在: {folder_path}")
        else:
            valid_folders.append(folder_path)
    
    if not valid_folders:
        print("没有找到有效的文件夹路径")
        return None
    
    print(f"开始处理 {len(valid_folders)} 个文件夹...")
    
    # 处理文件夹
    all_results = processor.process_multiple_folders(valid_folders, "./0125-part2/folder_results")
    
    print(f"\n文件夹处理完成！")
    print(f"共处理了 {len(all_results)} 个文件夹")
    
    # 统计总图片数量
    total_images = sum(len(folder_results) for folder_results in all_results.values())
    print(f"总共处理了 {total_images} 张图片")
    
    return all_results

if __name__ == "__main__":
    # 指定要处理的文件夹路径列表
    folder_paths = [
        "0125-part2\\figma",       
        "0125-part2\\guigpt",
        "0125-part2\\layoutcoder_fullpage",
        "0125-part2\\layoutcoder-original",
        "0125-part2\\screenshottocode",
    ]
    
    print("开始处理多个文件夹中的图片...")
    
    # 处理文件夹
    print("\n" + "="*60)
    print("处理多个文件夹...")
    print("="*60)
    
    # 过滤掉不存在的文件夹
    valid_folders = [folder for folder in folder_paths if os.path.exists(folder)]
    
    if valid_folders:
        process_multiple_folders_directly(valid_folders)
    else:
        print("❌ 没有找到有效的文件夹路径，请检查路径设置")
    
    print("\n处理完成！")