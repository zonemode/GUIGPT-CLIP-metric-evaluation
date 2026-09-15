import os
import numpy as np
from PIL import Image
import torch
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    AutoModelForCausalLM
)

def load_fg_clip2_model():
    """加载FG-CLIP2模型"""
    model_root = "C:/Users/king/.cache/huggingface/hub/fg-clip2-base"
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(model_root, trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 加载tokenizer和image_processor
    tokenizer = AutoTokenizer.from_pretrained(model_root)
    image_processor = AutoImageProcessor.from_pretrained(model_root)
    
    return model, image_processor, tokenizer, device

# 定义动态patch数量计算函数
def determine_max_value(image):
    w, h = image.size
    max_val = (w // 16) * (h // 16)  # 假设patch大小为16x16
    if max_val > 784:
        return 1024
    elif max_val > 576:
        return 784
    elif max_val > 256:
        return 576
    elif max_val > 128:
        return 256
    else:
        return 128

def compute_fg_clip2_similarity(image1_path, image2_path, model, image_processor, device):
    """计算两张图像的CLIP-I"""
    
    # 加载图像
    image1 = Image.open(image1_path).convert("RGB")
    image2 = Image.open(image2_path).convert("RGB")
    
    # 使用处理器进行预处理
    image_input1 = image_processor(
        images=image1, 
        max_num_patches=determine_max_value(image1), 
        return_tensors="pt"
    ).to(device)
    
    image_input2 = image_processor(
        images=image2, 
        max_num_patches=determine_max_value(image2), 
        return_tensors="pt"
    ).to(device)
    
    # 特征提取
    with torch.no_grad():
        image_features1 = model.get_image_features(**image_input1)
        image_features2 = model.get_image_features(**image_input2)
    
    # 特征归一化
    image_features1 = image_features1 / image_features1.norm(p=2, dim=-1, keepdim=True)
    image_features2 = image_features2 / image_features2.norm(p=2, dim=-1, keepdim=True)
    
    # 获取模型的缩放参数和偏置参数
    logit_scale, logit_bias = model.logit_scale.to(device), model.logit_bias.to(device)
    
    # 计算两种相似度（CLIP-I）
    # similarity1: 经过缩放和偏置调整的相似度（调整CLIP-I）
    similarity1 = (image_features1 @ image_features2.T) * logit_scale.exp() + logit_bias
    # similarity2: 原始余弦相似度（原始CLIP-I）
    similarity2 = (image_features1 @ image_features2.T)
    
    return similarity1.item(), similarity2.item()

# ====== 配置路径 ======
# 从当前脚本位置计算相对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.join(script_dir, '..', '..', 'Image Dataset', '0123')

# 原始图像路径（参考图像）
input_folder = os.path.join(base_path, 'Reference') 

# 对比图像路径（生成图像）
output_folders = [
    os.path.join(base_path, 'figma'),
    os.path.join(base_path, 'guigpt'),
    os.path.join(base_path, 'layoutcoderresult', 'image')
]

# 路径名称映射（用于结果显示）
folder_names = {
    os.path.join(base_path, 'figma'): 'Figma',
    os.path.join(base_path, 'guigpt'): 'Guigpt',
    os.path.join(base_path, 'layoutcoderresult', 'image'): 'Layoutcoder'
}

# ====== 加载模型 ======
print("正在加载CLIP模型...")
model, image_processor, tokenizer, device = load_fg_clip2_model()
print(f"模型已加载到设备: {device}")

# ====== 获取原始图像列表 ======
input_images = {}
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # 提取文件名（不含扩展名）作为键
        name = os.path.splitext(filename)[0]
        input_images[name] = os.path.join(input_folder, filename)

print(f"找到 {len(input_images)} 张原始图像")

# ====== 处理每个对比图像文件夹 ======
all_results = {}

for output_folder in output_folders:
    print(f"\n正在处理文件夹: {output_folder}")
    
    # 检查文件夹是否存在
    if not os.path.exists(output_folder):
        print(f"文件夹不存在: {output_folder}，跳过")
        continue
    
    # 获取对比图像列表
    output_images = {}
    for filename in os.listdir(output_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # 提取文件名（不含扩展名）作为键
            name = os.path.splitext(filename)[0]

            # 特殊处理layoutcoderresult文件夹的文件名
            if output_folder == os.path.join(base_path, 'layoutcoderresult', 'image'):
                # 从 "1_sep.html" 中提取 "1"
                if '_sep.html' in name:
                    name = name.split('_sep.html')[0]

            output_images[name] = os.path.join(output_folder, filename)
    
    print(f"找到 {len(output_images)} 张对比图像")
    
    # 找出共同的文件（基于文件名匹配）
    common_files = set(input_images.keys()) & set(output_images.keys())
    print(f"找到 {len(common_files)} 对匹配的图像")
    
    if not common_files:
        print("没有找到匹配的图像对，跳过此文件夹")
        continue
    
    # 计算每对图像的CLIP-I
    similarities1 = []  # 缩放偏置调整后的CLIP-I
    similarities2 = []  # 原始CLIP-I
    file_results = {}
    
    for filename in sorted(common_files, key=lambda x: int(x) if x.isdigit() else x):
        try:
            similarity1, similarity2 = compute_fg_clip2_similarity(
                input_images[filename],
                output_images[filename],
                model, image_processor, device
            )
            similarities1.append(similarity1)
            similarities2.append(similarity2)
            file_results[filename] = {
                'similarity1': similarity1,
                'similarity2': similarity2
            }
            print(f"{filename}: 调整CLIP-I={similarity1:.4f}, 原始CLIP-I={similarity2:.4f}")
            
        except Exception as e:
            print(f"处理 {filename} 出错: {e}")
    
    # 计算总体统计
    if similarities1 and similarities2:
        # 调整CLIP-I的统计
        avg_similarity1 = np.mean(similarities1)
        std_similarity1 = np.std(similarities1)
        max_similarity1 = np.max(similarities1)
        min_similarity1 = np.min(similarities1)
        
        # 原始CLIP-I的统计
        avg_similarity2 = np.mean(similarities2)
        std_similarity2 = np.std(similarities2)
        max_similarity2 = np.max(similarities2)
        min_similarity2 = np.min(similarities2)
        
        folder_name = folder_names.get(output_folder, os.path.basename(output_folder))
        all_results[output_folder] = {
            'adjusted': {
                'avg': avg_similarity1,
                'std': std_similarity1,
                'max': max_similarity1,
                'min': min_similarity1
            },
            'original': {
                'avg': avg_similarity2,
                'std': std_similarity2,
                'max': max_similarity2,
                'min': min_similarity2
            },
            'count': len(similarities1),
            'file_results': file_results
        }
        
        print(f"\n{folder_name} 结果")
        print(f"调整CLIP-I")
        print(f"平均: {avg_similarity1:.4f} ± {std_similarity1:.4f}")
        print(f"范围: {min_similarity1:.4f} - {max_similarity1:.4f}")
        print(f"\n原始CLIP-I")
        print(f"平均: {avg_similarity2:.4f} ± {std_similarity2:.4f}")
        print(f"范围: {min_similarity2:.4f} - {max_similarity2:.4f}")
        print(f"有效图像对: {len(similarities1)}/{len(common_files)}")
    else:
        print(f"无法计算 {output_folder} 的CLIP-I")

# ====== 生成比较报告 ======
print("\n" + "="*80)
print("CLIP-I比较报告")
print("="*80)

# 显示每个文件夹的结果
for output_folder in output_folders:
    if output_folder in all_results:
        result = all_results[output_folder]
        folder_name = folder_names.get(output_folder, os.path.basename(output_folder))
        print(f"\n{folder_name}:")
        print(f"调整CLIP-I: {result['adjusted']['avg']:.4f} ± {result['adjusted']['std']:.4f}")
        print(f"范围: {result['adjusted']['min']:.4f} - {result['adjusted']['max']:.4f}")
        print(f"\n原始CLIP-I: {result['original']['avg']:.4f} ± {result['original']['std']:.4f}")
        print(f"范围: {result['original']['min']:.4f} - {result['original']['max']:.4f}")
        print(f"有效图像对: {result['count']}")

# 比较结果（分别按调整CLIP-I和原始CLIP-I排序）
if len(all_results) >= 2:
    # 按调整CLIP-I排序
    print("\n" + "="*80)
    print("按调整CLIP-I排序的性能比较")
    print("="*80)
    
    sorted_results_adjusted = sorted(all_results.items(), 
                                   key=lambda x: x[1]['adjusted']['avg'], 
                                   reverse=True)
    
    for i, (folder, result) in enumerate(sorted_results_adjusted):
        folder_name = folder_names.get(folder, os.path.basename(folder))
        rank = i + 1
        print(f"{rank}. {folder_name}: {result['adjusted']['avg']:.4f} ± {result['adjusted']['std']:.4f}")
    
    # 按原始CLIP-I排序
    print("\n" + "="*80)
    print("按原始CLIP-I排序的性能比较")
    print("="*80)
    
    sorted_results_original = sorted(all_results.items(), 
                                   key=lambda x: x[1]['original']['avg'], 
                                   reverse=True)
    
    for i, (folder, result) in enumerate(sorted_results_original):
        folder_name = folder_names.get(folder, os.path.basename(folder))
        rank = i + 1
        print(f"{rank}. {folder_name}: {result['original']['avg']:.4f} ± {result['original']['std']:.4f}")

# 保存详细结果到文件
output_file = os.path.join(script_dir, '..', '..', 'Results', 'Fgclip2','Chinese_cuda_fg_CLIP-I_results_0123.txt')
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("CLIP-I 结果（调整CLIP-I + 原始CLIP-I）\n")
    f.write("="*70 + "\n\n")
    
    for output_folder in output_folders:
        if output_folder in all_results:
            result = all_results[output_folder]
            folder_name = folder_names.get(output_folder, os.path.basename(output_folder))
            f.write(f"文件夹: {folder_name}\n")
            f.write(f"平均调整CLIP-I: {result['adjusted']['avg']:.4f} ± {result['adjusted']['std']:.4f}\n")
            f.write(f"范围: {result['adjusted']['min']:.4f} - {result['adjusted']['max']:.4f}\n")
            f.write(f"平均原始CLIP-I: {result['original']['avg']:.4f} ± {result['original']['std']:.4f}\n")
            f.write(f"范围: {result['original']['min']:.4f} - {result['original']['max']:.4f}\n")
            f.write(f"有效图像对: {result['count']}\n\n")
            
            # 写入每对图像的详细结果
            f.write("各图像对CLIP-I:\n")
            f.write("文件名\t调整CLIP-I\t原始CLIP-I\n")
            f.write("-"*50 + "\n")
            for filename in sorted(result['file_results'].keys(), key=lambda x: int(x) if x.isdigit() else x):
                similarity1 = result['file_results'][filename]['similarity1']
                similarity2 = result['file_results'][filename]['similarity2']
                f.write(f"{filename}\t\t{similarity1:.4f}\t\t{similarity2:.4f}\n")
            
            f.write("\n" + "="*70 + "\n\n")

print(f"\n详细结果已保存到: {output_file}")