"""
数据下载脚本
Data Download Script

该脚本用于下载词义消歧(WSD)和语义角色标注(SRL)所需的数据集。
This script downloads datasets required for WSD and SRL tasks.
"""

import os
import argparse
import logging
from pathlib import Path
from typing import Optional

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_nltk_data():
    """
    下载NLTK所需的数据包
    Download required NLTK data packages
    """
    import nltk
    
    packages = [
        'wordnet',      # WordNet词汇数据库
        'omw-1.4',      # Open Multilingual Wordnet
        'semcor',       # SemCor语料库 (WSD标注数据)
        'stopwords',    # 停用词表
        'punkt',        # 句子分词器
        'averaged_perceptron_tagger',  # 词性标注器
        'brown',        # Brown语料库
    ]
    
    logger.info("开始下载NLTK数据包...")
    for package in packages:
        try:
            logger.info(f"下载 {package}...")
            nltk.download(package, quiet=True)
            logger.info(f"✓ {package} 下载成功")
        except Exception as e:
            logger.warning(f"✗ {package} 下载失败: {e}")
    
    logger.info("NLTK数据包下载完成")


def download_spacy_model(model_name: str = "en_core_web_sm"):
    """
    下载spaCy模型
    Download spaCy model
    
    Args:
        model_name: spaCy模型名称
    """
    import subprocess
    
    logger.info(f"开始下载spaCy模型: {model_name}")
    try:
        subprocess.run(
            ["python", "-m", "spacy", "download", model_name],
            check=True,
            capture_output=True
        )
        logger.info(f"✓ {model_name} 下载成功")
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {model_name} 下载失败: {e}")
        raise


def create_data_directories(base_path: str = "data"):
    """
    创建数据目录结构
    Create data directory structure
    
    Args:
        base_path: 基础数据路径
    """
    directories = [
        "raw/wsd/semcor",
        "raw/wsd/senseval",
        "raw/srl/propbank",
        "raw/srl/conll",
        "processed/wsd",
        "processed/srl",
        "cache",
    ]
    
    base = Path(base_path)
    for dir_path in directories:
        full_path = base / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"创建目录: {full_path}")
    
    logger.info("数据目录结构创建完成")


def download_wsd_data(output_dir: str = "data"):
    """
    下载WSD相关数据
    Download WSD-related data
    
    Args:
        output_dir: 输出目录
    """
    logger.info("="*50)
    logger.info("开始下载WSD数据...")
    logger.info("="*50)
    
    # 下载NLTK数据
    download_nltk_data()
    
    # 创建示例数据文件
    sample_data_path = Path(output_dir) / "raw/wsd/sample_data.txt"
    sample_data = """# WSD示例数据 / WSD Sample Data
# 格式: 句子 ||| 目标词 ||| 位置 ||| 词义ID

I went to the bank to deposit money. ||| bank ||| 4 ||| bank.n.01
The river bank was covered with flowers. ||| bank ||| 2 ||| bank.n.02
She will bank on his support. ||| bank ||| 2 ||| bank.v.01
The bright star is visible tonight. ||| star ||| 2 ||| star.n.01
The movie star attended the premiere. ||| star ||| 2 ||| star.n.04
He opened the window to let in fresh air. ||| window ||| 3 ||| window.n.01
Close all browser windows before restarting. ||| window ||| 3 ||| window.n.04
"""
    
    with open(sample_data_path, 'w', encoding='utf-8') as f:
        f.write(sample_data)
    logger.info(f"✓ 示例WSD数据已保存到 {sample_data_path}")
    
    logger.info("WSD数据下载完成")


def download_srl_data(output_dir: str = "data"):
    """
    下载SRL相关数据
    Download SRL-related data
    
    Args:
        output_dir: 输出目录
    """
    logger.info("="*50)
    logger.info("开始下载SRL数据...")
    logger.info("="*50)
    
    # 下载spaCy模型
    download_spacy_model()
    
    # 创建示例数据文件
    sample_data_path = Path(output_dir) / "raw/srl/sample_data.txt"
    sample_data = """# SRL示例数据 / SRL Sample Data
# CoNLL格式: 词 词性 谓词标记 角色标注

# 句子1: The cat ate the fish.
The	DT	-	B-ARG0
cat	NN	-	I-ARG0
ate	VBD	ate	B-V
the	DT	-	B-ARG1
fish	NN	-	I-ARG1
.	.	-	O

# 句子2: She gave him a book yesterday.
She	PRP	-	B-ARG0
gave	VBD	gave	B-V
him	PRP	-	B-ARG2
a	DT	-	B-ARG1
book	NN	-	I-ARG1
yesterday	RB	-	B-ARGM-TMP
.	.	-	O

# 句子3: The teacher asked the students to read the book.
The	DT	-	B-ARG0
teacher	NN	-	I-ARG0
asked	VBD	asked	B-V
the	DT	-	B-ARG1
students	NNS	-	I-ARG1
to	TO	-	B-ARG2
read	VB	-	I-ARG2
the	DT	-	I-ARG2
book	NN	-	I-ARG2
.	.	-	O
"""
    
    with open(sample_data_path, 'w', encoding='utf-8') as f:
        f.write(sample_data)
    logger.info(f"✓ 示例SRL数据已保存到 {sample_data_path}")
    
    logger.info("SRL数据下载完成")


def main():
    """主函数 / Main function"""
    parser = argparse.ArgumentParser(
        description="语义分析数据下载工具 / Semantic Analysis Data Download Tool"
    )
    parser.add_argument(
        "--wsd", 
        action="store_true", 
        help="下载WSD数据 / Download WSD data"
    )
    parser.add_argument(
        "--srl", 
        action="store_true", 
        help="下载SRL数据 / Download SRL data"
    )
    parser.add_argument(
        "--all", 
        action="store_true", 
        help="下载所有数据 / Download all data"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="data",
        help="输出目录 / Output directory (default: data)"
    )
    
    args = parser.parse_args()
    
    # 如果没有指定任何选项，默认下载全部
    if not any([args.wsd, args.srl, args.all]):
        args.all = True
    
    # 创建目录结构
    create_data_directories(args.output_dir)
    
    # 下载数据
    if args.all or args.wsd:
        download_wsd_data(args.output_dir)
    
    if args.all or args.srl:
        download_srl_data(args.output_dir)
    
    logger.info("="*50)
    logger.info("所有数据下载完成!")
    logger.info("="*50)


if __name__ == "__main__":
    main()
