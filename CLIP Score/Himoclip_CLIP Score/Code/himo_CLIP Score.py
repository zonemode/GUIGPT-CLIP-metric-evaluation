from model._HiMo_CLIP.model import himo
import torch
from PIL import Image
import json
import numpy as np
from packaging import version
import sklearn.preprocessing
import warnings
import collections
import os
import glob
import tiktoken  # 新增导入

class CLIPCapDataset(torch.utils.data.Dataset):
    """与文档一保持一致的数据集类"""
    def __init__(self, captions, prefix='A photo depicts:'):
        self.captions = captions
        self.prefix = prefix
        # 修复：检查前缀是否为空字符串
        if self.prefix and self.prefix[-1] != ' ':
            self.prefix += ' '

    def __getitem__(self, idx):
        caption = self.captions[idx]
        # 使用himo的tokenize替代clip.tokenize
        # 注意：这里不再添加前缀，因为文本已经包含了前缀
        tokenized = himo.tokenize(caption, truncate=True).squeeze()
        return {'caption': tokenized}

    def __len__(self):
        return len(self.captions)

def extract_all_captions(captions, model, device, batch_size=256, num_workers=0):
    """与文档一完全一致的文本特征提取函数"""
    data = torch.utils.data.DataLoader(
        CLIPCapDataset(captions, prefix=''),  # 前缀设为空，因为文本已经包含前缀
        batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
    all_text_features = []
    with torch.no_grad():
        for batch in data:          
            batch_captions = batch['caption'].to(device)
            text_features = model.encode_text(batch_captions)
            all_text_features.append(text_features.cpu().numpy())
    
    all_text_features = np.vstack(all_text_features)
    return all_text_features

def extract_all_images(image_paths, model, device, batch_size=64, num_workers=0):
    """与文档一完全一致的图像特征提取函数"""
    # 使用文档一的预处理流程
    def _transform_test(n_px):
        from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    
    class CLIPImageDataset(torch.utils.data.Dataset):
        def __init__(self, image_paths):
            self.image_paths = image_paths
            self.preprocess = _transform_test(224)
        
        def __getitem__(self, idx):
            image_path = self.image_paths[idx]
            image = Image.open(image_path)
            image = self.preprocess(image)
            return {'image': image}
        
        def __len__(self):
            return len(self.image_paths)
    
    data = torch.utils.data.DataLoader(
        CLIPImageDataset(image_paths),
        batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
    all_image_features = []
    with torch.no_grad():
        for batch in data:            
            batch_images = batch['image'].to(device)
            image_features = model.encode_image(batch_images)
            all_image_features.append(image_features.cpu().numpy())
    
    all_image_features = np.vstack(all_image_features)
    return all_image_features

def get_clip_score(model, images, candidates, device, w=2.5):
    """
    严格遵循文档一的CLIPScore计算算法
    """
    # 如果images是路径列表，先提取特征
    if isinstance(images, list):
        images = extract_all_images(images, model, device)
    
    # 提取文本特征
    candidates = extract_all_captions(candidates, model, device)
    
    # 归一化处理（与文档一完全一致）
    if version.parse(np.__version__) < version.parse('1.21'):
        images = sklearn.preprocessing.normalize(images, axis=1)
        candidates = sklearn.preprocessing.normalize(candidates, axis=1)
    else:
        warnings.warn(
            'due to a numerical instability, new numpy normalization is slightly different than paper results. '
            'to exactly replicate paper results, please use numpy version less than 1.21, e.g., 1.20.3.')
        images = images / np.sqrt(np.sum(images**2, axis=1, keepdims=True))
        candidates = candidates / np.sqrt(np.sum(candidates**2, axis=1, keepdims=True))
    
    # 计算相似度并应用CLIPScore公式
    similarities = np.sum(images * candidates, axis=1)
    per_instance_scores = w * np.clip(similarities, 0, None)
    mean_score = np.mean(per_instance_scores)
    
    return mean_score, per_instance_scores, candidates

def extract_text_from_json(json_data, prefix='A photo depicts:'):
    """从JSON数据中提取文本内容，并用三引号包裹"""
    contents = [text['content'] for text in json_data['texts']]
    combined_text = ' '.join(contents)
    
    # 添加提示词
    if prefix and prefix[-1] != ' ':  # 修复：检查prefix是否为空
        prefix += ' '
    full_text = prefix + combined_text
    
    # 用三引号包裹
    return f"'''{full_text}'''"

def count_tokens(text):
    """使用文档一的方法计算CLIP tokens数量"""
    # 使用tiktoken库计算CLIP tokens（GPT-2编码器）
    encoding = tiktoken.get_encoding("gpt2")
    tokens = encoding.encode(text)
    return len(tokens)

def preprocess_json_files(json_files):
    """预处理所有JSON文件，计算token数量并筛选"""
    valid_json_files = []
    skipped_files = []
    
    for json_file in json_files:
        # 提取JSON文件的基本名（不带扩展名）
        json_id = os.path.splitext(os.path.basename(json_file))[0]
        
        # 加载JSON数据并计算token数量
        try:
            with open(json_file, 'r') as f:
                json_data = json.load(f)
            
            # 提取文本并计算token数量
            candidate_text = extract_text_from_json(json_data)
            token_count = count_tokens(candidate_text)
            
            # 如果token数量大于248，跳过这个文件
            if token_count > 248:
                print(f"跳过 {json_id}.json: token数量 {token_count} > 248")
                skipped_files.append((json_id, token_count))
            else:
                valid_json_files.append(json_file)
                
        except Exception as e:
            print(f"处理文件 {json_file} 时出错: {e}")
            skipped_files.append((json_id, "处理错误"))
    
    return valid_json_files, skipped_files

def find_matching_images(json_files, image_folder):
    """在指定图片文件夹中查找匹配的图片"""
    image_paths = []
    json_ids = []
    
    for json_file in json_files:
        # 提取JSON文件的基本名（不带扩展名）
        json_id = os.path.splitext(os.path.basename(json_file))[0]
        
        # 尝试多种可能的图片命名模式
        patterns = [
            f"{json_id}.png",           # 标准格式：1.png
            f"{json_id}.jpg",           # 标准格式：1.jpg
            f"{json_id}.jpeg",          # 标准格式：1.jpeg
            f"{json_id}_sep_fullpage.png", # layoutcoder_fullpage格式
            f"{json_id}_sep.html.png",   # layoutcoder-original格式
            f"{json_id}_*",             # 其他可能的变体
        ]
        
        found = False
        for pattern in patterns:
            matches = glob.glob(os.path.join(image_folder, pattern))
            if matches:
                image_paths.append(matches[0])  # 使用第一个匹配项
                json_ids.append(json_id)
                print(f"为JSON文件 {json_id} 找到匹配图片: {matches[0]}")
                found = True
                break
        
        if not found:
            print(f"警告: 未找到与JSON文件 {json_id} 匹配的图片")
    
    return json_ids, image_paths

def get_image_id_from_filename(filename):
    """从图片文件名中提取ID"""
    # 处理不同的命名格式
    base_name = os.path.splitext(filename)[0]
    
    # 处理_layoutcoder_fullpage格式
    if base_name.endswith('_sep_fullpage'):
        return base_name.replace('_sep_fullpage', '')
    
    # 处理_layoutcoder-original格式
    if base_name.endswith('_sep.html'):
        return base_name.replace('_sep.html', '')
    
    # 标准格式，直接返回
    return base_name

def process_image_folder(image_folder, valid_json_files, all_skipped_files, model, device, result_dir):
    """处理单个图片文件夹"""
    print(f"\n处理图片文件夹: {image_folder}")
    
    # 查找匹配的图片
    json_ids, image_paths = find_matching_images(valid_json_files, image_folder)
    
    # 获取图片文件夹中的所有图片文件
    all_image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total_images = len(all_image_files)
    
    if not image_paths:
        print(f"在 {image_folder} 中没有找到有效的JSON-图片对")
        # 即使没有匹配的图片，也要创建结果文件
        folder_name = os.path.basename(image_folder)
        folder_result_dir = os.path.join(result_dir, folder_name)
        os.makedirs(folder_result_dir, exist_ok=True)
        
        # 创建结果文件
        result_file = os.path.join(folder_result_dir, "results.txt")
        
        # 写入结果文件
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write("图片名称和CLIP-Score结果:\n")
            f.write("=" * 50 + "\n")
            f.write("无有效结果\n\n")
            
            # 统计信息
            f.write("统计信息:\n")
            f.write("=" * 50 + "\n")
            f.write(f"图片总数: {total_images}\n")
            f.write(f"有CLIP-Score结果的图片数量: 0\n")
            f.write(f"无CLIP-Score结果的图片数量: {total_images}\n\n")
            
            # 无CLIP-Score结果的图片列表
            f.write("无CLIP-Score结果的图片列表:\n")
            f.write("=" * 50 + "\n")
            for image_file in all_image_files:
                image_id = get_image_id_from_filename(image_file)
                # 检查是否有对应的JSON文件
                json_file_exists = any(os.path.splitext(os.path.basename(f))[0] == image_id for f in valid_json_files)
                skipped_info = next(((sid, tcount) for sid, tcount in all_skipped_files if sid == image_id), None)
                
                if skipped_info:
                    f.write(f"{image_file}: JSON文件被跳过 (token数量: {skipped_info[1]})\n")
                elif not json_file_exists:
                    f.write(f"{image_file}: 没有对应的JSON文件\n")
                else:
                    f.write(f"{image_file}: 其他原因\n")
        
        print(f"结果已保存到: {result_file}")
        return None
    
    # 提取所有JSON文件中的文本
    candidate_texts = []
    valid_json_ids = []
    
    for json_file in valid_json_files:
        json_id = os.path.splitext(os.path.basename(json_file))[0]
        if json_id in json_ids:  # 只处理有匹配图片的JSON文件
            with open(json_file, 'r') as f:
                json_data = json.load(f)
            candidate_text = extract_text_from_json(json_data)
            candidate_texts.append(candidate_text)
            valid_json_ids.append(json_id)
    
    # 使用与文档一完全相同的接口计算CLIPScore
    mean_score, per_instance_scores, candidate_features = get_clip_score(
        model, 
        image_paths,  # 图像路径列表
        candidate_texts,  # 文本列表
        device,
        w=2.5  # 与文档一相同的默认权重
    )
    
    # 创建结果文件夹
    folder_name = os.path.basename(image_folder)
    folder_result_dir = os.path.join(result_dir, folder_name)
    os.makedirs(folder_result_dir, exist_ok=True)
    
    # 创建结果文件
    result_file = os.path.join(folder_result_dir, "results.txt")
    
    # 计算统计信息
    processed_images = len(per_instance_scores)
    max_score = np.max(per_instance_scores) if processed_images > 0 else 0
    min_score = np.min(per_instance_scores) if processed_images > 0 else 0
    avg_score = np.mean(per_instance_scores) if processed_images > 0 else 0
    
    # 找到最大和最小分数对应的图片
    max_score_index = np.argmax(per_instance_scores) if processed_images > 0 else -1
    min_score_index = np.argmin(per_instance_scores) if processed_images > 0 else -1
    
    max_score_image = os.path.basename(image_paths[max_score_index]) if max_score_index != -1 else "N/A"
    min_score_image = os.path.basename(image_paths[min_score_index]) if min_score_index != -1 else "N/A"
    
    # 获取有CLIP-Score结果的图片文件名
    processed_image_files = [os.path.basename(path) for path in image_paths]
    
    # 获取无CLIP-Score结果的图片文件名
    unprocessed_image_files = [f for f in all_image_files if f not in processed_image_files]
    
    # 写入结果文件
    with open(result_file, 'w', encoding='utf-8') as f:
        # ① 对应的图片文件夹中能与json文件对应的图片的图片名字和clip-score结果
        f.write("图片名称和CLIP-Score结果:\n")
        f.write("=" * 50 + "\n")
        for i, (json_id, score) in enumerate(zip(valid_json_ids, per_instance_scores)):
            image_name = os.path.basename(image_paths[i])
            f.write(f"{image_name}: {score:.4f}\n")
        
        f.write("\n")
        
        # ② 对应的图片文件夹中图片的总数和有clip-score结果的图片总数
        f.write("统计信息:\n")
        f.write("=" * 50 + "\n")
        f.write(f"图片总数: {total_images}\n")
        f.write(f"有CLIP-Score结果的图片数量: {processed_images}\n")
        f.write(f"无CLIP-Score结果的图片数量: {len(unprocessed_image_files)}\n\n")
        
        # ③ 最大的clip-score和最小的clip-score，以及它们分别对应哪张图片
        f.write(f"最大CLIP-Score: {max_score:.4f} (对应图片: {max_score_image})\n")
        f.write(f"最小CLIP-Score: {min_score:.4f} (对应图片: {min_score_image})\n")
        
        # ④ 对应的图片文件夹的clip-score范围和平均数
        f.write(f"CLIP-Score范围: {min_score:.4f} - {max_score:.4f}\n")
        f.write(f"平均CLIP-Score: {avg_score:.4f}\n\n")
        
        # ⑤ 无CLIP-Score结果的图片列表及原因
        f.write("无CLIP-Score结果的图片列表:\n")
        f.write("=" * 50 + "\n")
        
        if unprocessed_image_files:
            for image_file in unprocessed_image_files:
                image_id = get_image_id_from_filename(image_file)
                # 检查是否有对应的JSON文件
                json_file_exists = any(os.path.splitext(os.path.basename(f))[0] == image_id for f in valid_json_files)
                skipped_info = next(((sid, tcount) for sid, tcount in all_skipped_files if sid == image_id), None)
                
                if skipped_info:
                    f.write(f"{image_file}: JSON文件被跳过 (token数量: {skipped_info[1]})\n")
                elif not json_file_exists:
                    f.write(f"{image_file}: 没有对应的JSON文件\n")
                else:
                    f.write(f"{image_file}: 其他原因\n")
        else:
            f.write("所有图片都有CLIP-Score结果\n")
    
    print(f"结果已保存到: {result_file}")
    
    return {
        'folder_name': folder_name,
        'total_images': total_images,
        'processed_images': processed_images,
        'max_score': max_score,
        'min_score': min_score,
        'avg_score': avg_score,
        'max_score_image': max_score_image,
        'min_score_image': min_score_image
    }

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = himo.load("D:/study/anaconda3/envs/pytorch_cuda/Lib/site-packages/model/_HiMo_CLIP/weights/himo_clip_L.pt", device=device)
    model.eval()
    
    # 从当前脚本位置计算相对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(script_dir, '..', '..', 'Image Dataset')
    
    # JSON文件夹路径
    json_folder = os.path.join(base_path, '..', '探索clip-score', 'Reference')   
    
    # 图片文件夹路径列表
    image_folders = [
        os.path.join(base_path, 'Reference'),            # 标准图片文件夹
        os.path.join(base_path, 'figma'),                # 标准格式
        os.path.join(base_path, 'guigpt'),               # 标准格式
        os.path.join(base_path, 'layoutcoder_fullpage'), # _sep_fullpage格式
        os.path.join(base_path, 'layoutcoder-original'),  # _sep.html格式
        os.path.join(base_path, 'screenshottocode'),    # 标准格式
    ]
    
    # 创建最终结果文件夹
    result_dir = os.path.join(base_path, '..', 'Himoclip_CLIP Score', 'himo_clip-s_results')
    os.makedirs(result_dir, exist_ok=True)
    
    # 获取所有JSON文件
    json_files = glob.glob(os.path.join(json_folder, "*.json"))
    print(f"找到 {len(json_files)} 个JSON文件")
    
    # 预处理JSON文件，计算token数量并筛选
    valid_json_files, all_skipped_files = preprocess_json_files(json_files)
    print(f"有效JSON文件数量: {len(valid_json_files)}")
    print(f"跳过的JSON文件数量: {len(all_skipped_files)}")
    
    # 处理每个图片文件夹
    for image_folder in image_folders:
        if not os.path.exists(image_folder):
            print(f"警告: 图片文件夹 {image_folder} 不存在，跳过")
            continue
            
        folder_stats = process_image_folder(image_folder, valid_json_files, all_skipped_files, model, device, result_dir)
    
    # 创建跳过的文件记录（只输出一次）
    skipped_file = os.path.join(result_dir, "跳过的文件.txt")
    with open(skipped_file, 'w', encoding='utf-8') as f:
        f.write("跳过的JSON文件列表:\n")
        f.write("=" * 50 + "\n")
        for json_id, token_count in all_skipped_files:
            f.write(f"{json_id}.json: token数量 = {token_count}\n")
    
    print(f"跳过的文件记录已保存到: {skipped_file}")
    
    # 打印总体统计信息
    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)
    print(f"跳过的JSON文件总数: {len(all_skipped_files)}")
    
    # 检查每个结果文件夹的内容
    for image_folder in image_folders:
        folder_name = os.path.basename(image_folder)
        folder_result_dir = os.path.join(result_dir, folder_name)
        result_file = os.path.join(folder_result_dir, "results.txt")
        
        if os.path.exists(result_file):
            with open(result_file, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"\n{image_folder} 的结果:")
                # 只打印图片和分数部分
                parts = content.split("统计信息:")
                if len(parts) > 0:
                    print(parts[0])
    
    return all_skipped_files

if __name__ == '__main__':
    main()