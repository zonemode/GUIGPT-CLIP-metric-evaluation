#vit_0123

import os
import numpy as np
import clip
from PIL import Image
import torch

def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

def compute_clip_similarity(image1_path, image2_path, model, preprocess, device):
    """计算两张图像的CLIP-I"""
    # 加载并预处理图像
    image1 = preprocess(Image.open(image1_path)).unsqueeze(0).to(device)
    image2 = preprocess(Image.open(image2_path)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features1 = model.encode_image(image1)
        features2 = model.encode_image(image2)
    
    # 归一化
    features1 /= features1.norm(dim=-1, keepdim=True)
    features2 /= features2.norm(dim=-1, keepdim=True)
    
    # 计算余弦相似度（CLIP-I）
    similarity = (features1 @ features2.T).item()
    return similarity

# ====== 配置路径 ======
# 从当前脚本位置计算相对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.join(script_dir, '..', '..', 'Image Dataset', '0123')

# 原始图像路径（参考图像）
input_folder = os.path.join(base_path, 'Reference') 

# 对比图像路径（对比图像）
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
model, preprocess, device = load_clip_model()
print(f"模型已加载到设备上: {device}")

# ====== 获取原始图像列表 ======
input_images = {}
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # 提取文件名（不包括扩展名）作为键
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
    similarities = []
    file_results = {}
    
    for filename in sorted(common_files, key=lambda x: int(x) if x.isdigit() else x):
        try:
            similarity = compute_clip_similarity(
                input_images[filename],
                output_images[filename],
                model, preprocess, device
            )
            similarities.append(similarity)
            file_results[filename] = similarity
            print(f"{filename}: {similarity:.4f}")
            
        except Exception as e:
            print(f"处理 {filename} 出错: {e}")
    
    # 计算总体统计
    if similarities:
        avg_similarity = np.mean(similarities)
        std_similarity = np.std(similarities)
        max_similarity = np.max(similarities)
        min_similarity = np.min(similarities)
        
        folder_name = folder_names.get(output_folder, os.path.basename(output_folder))
        all_results[output_folder] = {
            'avg': avg_similarity,
            'std': std_similarity,
            'max': max_similarity,
            'min': min_similarity,
            'count': len(similarities),
            'file_results': file_results
        }
        
        print(f"\n{folder_name} 结果")
        print(f"平均CLIP-I: {avg_similarity:.4f}")
        print(f"标准差: {std_similarity:.4f}")
        print(f"最高CLIP-I: {max_similarity:.4f}")
        print(f"最低CLIP-I: {min_similarity:.4f}")
        print(f"有效图像对: {len(similarities)}/{len(common_files)}")
    else:
        print(f"无法计算 {output_folder} 的CLIP-I")

# ====== 生成比较报告 ======
print("\n" + "="*60)
print("CLIP-I比较报告")
print("="*60)

# 显示每个文件夹的结果
for output_folder in output_folders:
    if output_folder in all_results:
        result = all_results[output_folder]
        folder_name = folder_names.get(output_folder, os.path.basename(output_folder))
        print(f"\n{folder_name}:")
        print(f"平均CLIP-I: {result['avg']:.4f} ± {result['std']:.4f}")
        print(f"范围: {result['min']:.4f} - {result['max']:.4f}")
        print(f"有效图像对: {result['count']}")

# 比较结果
if len(all_results) >= 2:
    print("\n" + "="*60)
    print("性能比较")
    print("="*60)
    
    # 按CLIP-I排序
    sorted_results = sorted(all_results.items(), 
                           key=lambda x: x[1]['avg'], 
                           reverse=True)
    
    for i, (folder, result) in enumerate(sorted_results):
        folder_name = folder_names.get(folder, os.path.basename(folder))
        rank = i + 1
        print(f"{rank}. {folder_name}: {result['avg']:.4f} ± {result['std']:.4f}")

# 保存详细结果到文件
output_file = os.path.join(script_dir, '..', '..', 'Results', 'Vit-B32','Chinese_cuda_vit_CLIP-I_results_0123.txt')
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("CLIP-I 结果\n")
    f.write("="*50 + "\n\n")
    
    for output_folder in output_folders:
        if output_folder in all_results:
            result = all_results[output_folder]
            folder_name = folder_names.get(output_folder, os.path.basename(output_folder))
            f.write(f"文件夹: {folder_name}\n")
            f.write(f"平均CLIP-I: {result['avg']:.4f} ± {result['std']:.4f}\n")
            f.write(f"范围: {result['min']:.4f} - {result['max']:.4f}\n")
            f.write(f"有效图像对: {result['count']}\n\n")
            
            # 写入每对图像的详细结果
            f.write("各图像对CLIP-I:\n")
            for filename in sorted(result['file_results'].keys(), key=lambda x: int(x) if x.isdigit() else x):
                similarity = result['file_results'][filename]
                f.write(f"{filename}: {similarity:.4f}\n")
            
            f.write("\n" + "-"*50 + "\n\n")

print(f"\n详细结果已保存到: {output_file}")