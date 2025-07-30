import os
import csv
import argparse
from pathlib import Path

def generate_modified_csv(root_dir, output_csv):
    """
    根据数据库根目录生成修改路径后的CSV文件
    
    参数:
    root_dir (str): 数据库根目录
    output_csv (str): 输出CSV文件的路径
    """
    # 解析根目录，确保路径正确
    root_path = Path(root_dir).resolve()

    
    # 检查目录是否存在
    if not root_path.exists() or not root_path.is_dir():
        raise ValueError(f"数据库目录不存在: {root_path}")
    
    # 收集所有符合条件的音频文件路径
    data_entries = []
    entry_id = 1  # ID从1开始递增
    
    # 遍历录制组号（如Z01、Z02等）
    for z_dir in root_path.iterdir():
        if not z_dir.is_dir() or not z_dir.name.startswith("Z"):
            continue  # 跳过非录制组目录
        
        # 遍历音频编号（如0001、0002等）
        for audio_dir in z_dir.iterdir():
            if not audio_dir.is_dir() or not audio_dir.name.isdigit():
                continue  # 跳过非音频编号目录
            
            # 构建各个文件的路径
            # 两个混合音频路径
            mix1_wav_path = audio_dir / "array" / "mixture-1.wav"
            mix2_wav_path = audio_dir / "array" / "mixture-2.wav"
            
            # 声源文件使用source文件夹下的source-1至source-4.wav
            s1_wav_path = audio_dir / "source" / "source-1.wav"
            s2_wav_path = audio_dir / "source" / "source-2.wav"
            s3_wav_path = audio_dir / "source" / "source-3.wav"
            s4_wav_path = audio_dir / "source" / "source-4.wav"
            
            # 检查文件是否存在
            required_files = [
                mix1_wav_path, mix2_wav_path, 
                s1_wav_path, s2_wav_path, s3_wav_path, s4_wav_path
            ]
            
            if all(file.exists() for file in required_files):
                # 构造一行数据（ID递增，duration固定为1，格式固定为wav）
                data_entries.append({
                    "ID": entry_id,
                    "duration": 1,
                    "mix1_wav": str(mix1_wav_path),
                    "mix1_wav_format": "wav",
                    "mix1_wav_opts": "",
                    "mix2_wav": str(mix2_wav_path),
                    "mix2_wav_format": "wav",
                    "mix2_wav_opts": "",
                    "s1_wav": str(s1_wav_path),
                    "s1_wav_format": "wav",
                    "s1_wav_opts": "",
                    "s2_wav": str(s2_wav_path),
                    "s2_wav_format": "wav",
                    "s2_wav_opts": "",
                    "s3_wav": str(s3_wav_path),
                    "s3_wav_format": "wav",
                    "s3_wav_opts": "",
                    "s4_wav": str(s4_wav_path),
                    "s4_wav_format": "wav",
                    "s4_wav_opts": ""
                })
                entry_id += 1
            else:
                # 打印警告但继续处理其他文件
                missing = [str(f) for f in required_files if not f.exists()]
                print(f"警告: 跳过不完整的音频目录 {audio_dir}，缺失文件: {missing}")
    
    # 写入CSV文件
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            "ID", "duration", 
            "mix1_wav", "mix1_wav_format", "mix1_wav_opts",
            "mix2_wav", "mix2_wav_format", "mix2_wav_opts",
            "s1_wav", "s1_wav_format", "s1_wav_opts",
            "s2_wav", "s2_wav_format", "s2_wav_opts",
            "s3_wav", "s3_wav_format", "s3_wav_opts",
            "s4_wav", "s4_wav_format", "s4_wav_opts"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data_entries)
    
    print(f"CSV文件生成完成: {output_csv}")
    print(f"共生成 {len(data_entries)} 条记录")

def main():
    parser = argparse.ArgumentParser(description='生成修改路径后的CSV文件')
    parser.add_argument('--root_dir', required=True, help='数据库根目录（包含development_set的路径）')
    parser.add_argument('--output_csv', required=True, help='输出CSV文件的路径')
    
    args = parser.parse_args()
    
    try:
        generate_modified_csv(args.root_dir, args.output_csv)
    except Exception as e:
        print(f"生成CSV时出错: {e}")

if __name__ == "__main__":
    main()