# 文本拼接方法：将json格式中每一个id的'content'段的内容提取出来拼在一起，之间用空格隔开。
import json
import tiktoken
import os
import glob
from datetime import datetime

def calculate_exact_clip_tokens(json_data):
    """
    使用tiktoken库计算CLIP tokens长度
    """
    # 加载GPT-2编码器（CLIP使用相同的tokenizer）
    encoding = tiktoken.get_encoding("gpt2")
    
    # 提取文本内容
    contents = []
    for text_item in json_data.get('texts', []):
        content = text_item.get('content', '').strip()
        if content:
            contents.append(content)
    
    combined_text = ' '.join(contents)
    tokens = encoding.encode(combined_text)
    
    return {
        'combined_text': combined_text,
        'tokens_length': len(tokens),
        'tokens': tokens,
        'word_count': len(combined_text.split()),
        'character_count': len(combined_text)
    }

def process_single_json_file(file_path):
    """
    处理单个JSON文件
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        result = calculate_exact_clip_tokens(json_data)
        result['file_name'] = os.path.basename(file_path)
        result['file_path'] = file_path  # 保存完整路径以便后续使用
        return result
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return None

def process_multiple_json_files(file_paths, output_txt_path):
    """
    处理多个JSON文件并将结果保存到TXT文件
    """
    all_results = []
    
    # 处理每个文件
    for file_path in file_paths:
        print(f"正在处理: {file_path}")
        result = process_single_json_file(file_path)
        if result:
            all_results.append(result)
    
    # 生成TXT文件内容
    txt_content = generate_txt_content(all_results)
    
    # 保存到TXT文件
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write(txt_content)
    
    print(f"结果已保存到: {output_txt_path}")
    return all_results

def generate_txt_content(results):
    """
    生成TXT文件内容
    """
    # 添加标题和时间戳
    content = f"CLIP Tokens长度分析报告\n"
    content += f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    content += f"处理文件数量: {len(results)}\n"
    content += "=" * 60 + "\n\n"
    
    # 添加每个文件的结果
    for i, result in enumerate(results, 1):
        content += f"文件 {i}: {result['file_name']}\n"
        content += f"合并文本: {result['combined_text']}\n"
        content += f"字符数: {result['character_count']}\n"
        content += f"单词数: {result['word_count']}\n"
        content += f"CLIP tokens长度: {result['tokens_length']}\n"
        content += f"Tokens列表: {result['tokens']}\n"
        content += "-" * 40 + "\n\n"
    
    # 计算最大和最小tokens
    if results:
        max_tokens_result = max(results, key=lambda x: x['tokens_length'])
        min_tokens_result = min(results, key=lambda x: x['tokens_length'])
        
        # 统计tokens大于248、196和77的文件
        tokens_gt_248 = [r for r in results if r['tokens_length'] > 248]
        tokens_gt_196 = [r for r in results if r['tokens_length'] > 196]
        tokens_gt_77 = [r for r in results if r['tokens_length'] > 77]
        
        # 添加汇总统计
        content += "汇总统计:\n"
        total_tokens = sum(result['tokens_length'] for result in results)
        total_chars = sum(result['character_count'] for result in results)
        total_words = sum(result['word_count'] for result in results)
        
        content += f"总tokens长度: {total_tokens}\n"
        content += f"总字符数: {total_chars}\n"
        content += f"总单词数: {total_words}\n"
        content += f"平均tokens长度: {total_tokens/len(results):.2f}\n"
        content += f"最大tokens长度: {max_tokens_result['tokens_length']} (文件: {max_tokens_result['file_name']})\n"
        content += f"最小tokens长度: {min_tokens_result['tokens_length']} (文件: {min_tokens_result['file_name']})\n"
        content += f"tokens范围: {min_tokens_result['tokens_length']} - {max_tokens_result['tokens_length']}\n"
        
        # 添加tokens大于248的统计
        content += f"\nTokens长度 > 248 的文件数量: {len(tokens_gt_248)}\n"
        if tokens_gt_248:
            content += "Tokens长度 > 248 的文件列表:\n"
            for i, result in enumerate(tokens_gt_248, 1):
                content += f"  {i}. {result['file_name']} (tokens长度: {result['tokens_length']})\n"
        else:
            content += "没有文件tokens长度 > 248\n"
        
        # 添加tokens大于196的统计
        content += f"\nTokens长度 > 196 的文件数量: {len(tokens_gt_196)}\n"
        if tokens_gt_196:
            content += "Tokens长度 > 196 的文件列表:\n"
            for i, result in enumerate(tokens_gt_196, 1):
                content += f"  {i}. {result['file_name']} (tokens长度: {result['tokens_length']})\n"
        else:
            content += "没有文件tokens长度 > 196\n"
        
        # 添加tokens大于77的统计
        content += f"\nTokens长度 > 77 的文件数量: {len(tokens_gt_77)}\n"
        if tokens_gt_77:
            content += "Tokens长度 > 77 的文件列表:\n"
            for i, result in enumerate(tokens_gt_77, 1):
                content += f"  {i}. {result['file_name']} (tokens长度: {result['tokens_length']})\n"
        else:
            content += "没有文件tokens长度 > 77\n"
    else:
        content += "汇总统计:\n"
        content += "没有找到有效的结果\n"
    
    return content

def find_json_files(directory):
    """
    在指定目录中查找所有JSON文件
    """
    pattern = os.path.join(directory, "*.json")
    return glob.glob(pattern)

# 主程序
if __name__ == "__main__":
    # 方式1: 直接指定文件列表
    # json_files = ["file1.json","file2.json",
        # 添加更多文件路径...]
    
    # 方式2: 自动查找目录中的所有JSON文件
    json_files = find_json_files("./0125-part2/探索clip-score/Reference")  # 替换为你的目录路径
    
    # 输出文件路径
    output_txt_path = "./0125-part2/clip_tokens_results.txt"
    
    # 处理所有JSON文件
    results = process_multiple_json_files(json_files, output_txt_path)
    
    # 在控制台也显示结果
    print("\n=== 处理完成 ===")
    
    if results:
        # 计算最大和最小tokens
        max_tokens_result = max(results, key=lambda x: x['tokens_length'])
        min_tokens_result = min(results, key=lambda x: x['tokens_length'])
        
        # 统计tokens大于248、196和77的文件
        tokens_gt_248 = [r for r in results if r['tokens_length'] > 248]
        tokens_gt_196 = [r for r in results if r['tokens_length'] > 196]
        tokens_gt_77 = [r for r in results if r['tokens_length'] > 77]
        
        for result in results:
            print(f"\n文件: {result['file_name']}")
            print(f"合并文本: {result['combined_text']}")
            print(f"字符数: {result['character_count']}")
            print(f"单词数: {result['word_count']}")
            print(f"CLIP tokens长度: {result['tokens_length']}")
        
        # 显示汇总信息
        print("\n=== 汇总信息 ===")
        print(f"处理文件总数: {len(results)}")
        print(f"最大tokens长度: {max_tokens_result['tokens_length']} (文件: {max_tokens_result['file_name']})")
        print(f"最小tokens长度: {min_tokens_result['tokens_length']} (文件: {min_tokens_result['file_name']})")
        print(f"tokens范围: {min_tokens_result['tokens_length']} - {max_tokens_result['tokens_length']}")
        
        # 计算并显示其他统计信息
        total_tokens = sum(result['tokens_length'] for result in results)
        print(f"总tokens长度: {total_tokens}")
        print(f"平均tokens长度: {total_tokens/len(results):.2f}")
        
        # 显示tokens大于248、196和77的统计
        print(f"\nTokens长度 > 248 的文件数量: {len(tokens_gt_248)}")
        if tokens_gt_248:
            print("Tokens长度 > 248 的文件列表:")
            for i, result in enumerate(tokens_gt_248, 1):
                print(f"  {i}. {result['file_name']} (tokens长度: {result['tokens_length']})")
        else:
            print("没有文件tokens长度 > 248")
        
        print(f"\nTokens长度 > 196 的文件数量: {len(tokens_gt_196)}")
        if tokens_gt_196:
            print("Tokens长度 > 196 的文件列表:")
            for i, result in enumerate(tokens_gt_196, 1):
                print(f"  {i}. {result['file_name']} (tokens长度: {result['tokens_length']})")
        else:
            print("没有文件tokens长度 > 196")
        
        print(f"\nTokens长度 > 77 的文件数量: {len(tokens_gt_77)}")
        if tokens_gt_77:
            print("Tokens长度 > 77 的文件列表:")
            for i, result in enumerate(tokens_gt_77, 1):
                print(f"  {i}. {result['file_name']} (tokens长度: {result['tokens_length']})")
        else:
            print("没有文件tokens长度 > 77")
    else:
        print("没有找到有效的结果")