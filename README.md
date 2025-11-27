# 语义分析模型：词义消歧与语义角色标注

本项目实现了词义消歧(Word Sense Disambiguation, WSD)和语义角色标注(Semantic Role Labeling, SRL)任务，包含多种方法的实现与对比分析。

## 📋 项目简介

### 词义消歧 (WSD)
词义消歧是确定多义词在特定上下文中具体含义的任务。本项目实现了以下方法：
- **Lesk算法**：基于词义定义与上下文词重叠的传统方法
- **BERT上下文方法**：使用预训练模型获取词的上下文嵌入表示
- **知识库增强方法**：结合WordNet图结构和语义相似度
- **GlossBERT方法**：联合编码上下文和词义定义

### 语义角色标注 (SRL)
语义角色标注是识别句子中谓词的论元及其语义角色的任务。本项目实现了：
- **基于句法树的方法**：利用依存句法分析映射语义角色
- **基于神经网络的方法**：使用BERT进行序列标注

## 🛠️ 环境配置

### 1. 克隆仓库
```bash
git clone https://github.com/linzhou03/nlp-project.git
cd nlp-project
```

### 2. 创建虚拟环境（推荐）
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 下载必要的模型和数据
```bash
# 下载NLTK数据
python -c "import nltk; nltk.download('wordnet'); nltk.download('semcor'); nltk.download('stopwords'); nltk.download('omw-1.4')"

# 下载spaCy模型
python -m spacy download en_core_web_sm
```

## 📁 项目结构

```
nlp-project/
├── README.md                    # 项目说明文档
├── requirements.txt             # Python依赖
├── main.py                      # 主程序入口
├── data/                        # 数据目录
│   └── README.md               # 数据下载说明
├── wsd/                         # 词义消歧模块
│   ├── __init__.py
│   ├── context_based.py        # 基于上下文的方法
│   └── knowledge_based.py      # 基于知识库的方法
├── srl/                         # 语义角色标注模块
│   ├── __init__.py
│   ├── syntax_based.py         # 基于句法树的方法
│   └── neural_based.py         # 基于神经网络的方法
├── evaluation/                  # 评估模块
│   ├── __init__.py
│   └── evaluator.py            # 评估器
├── utils/                       # 工具模块
│   ├── __init__.py
│   └── data_loader.py          # 数据加载器
├── experiments/                 # 实验脚本
│   ├── run_wsd.py              # WSD实验
│   ├── run_srl.py              # SRL实验
│   └── analysis.py             # 结果分析
└── report/                      # 实验报告
    └── experiment_report.md    # 实验报告模板
```

## 🚀 使用方法

### 运行词义消歧实验
```bash
# 运行所有WSD方法
python main.py --task wsd --method all

# 只运行Lesk算法
python main.py --task wsd --method lesk

# 只运行BERT上下文方法
python main.py --task wsd --method bert
```

### 运行语义角色标注实验
```bash
# 运行所有SRL方法
python main.py --task srl --method all

# 只运行句法树方法
python main.py --task srl --method syntax

# 只运行神经网络方法
python main.py --task srl --method neural
```

### 运行完整实验并生成报告
```bash
python main.py --task all --report
```

### 快速演示
```bash
# WSD演示
python -c "
from wsd.context_based import LeskAlgorithm
lesk = LeskAlgorithm()
result = lesk.disambiguate('I went to the bank to deposit money', 'bank')
print(f'词义: {result[0].definition()}')
"

# SRL演示
python -c "
from srl.syntax_based import SyntaxBasedSRL
srl = SyntaxBasedSRL()
srl.visualize('The cat chased the mouse in the garden')
"
```

## 📊 实验结果

### 词义消歧结果 (SemCor数据集)

| 方法 | 准确率 |
|------|--------|
| Lesk算法 | ~45% |
| BERT上下文 | ~65% |
| 知识库增强 | ~60% |
| GlossBERT | ~70% |

### 语义角色标注结果 (PropBank数据集)

| 方法 | Precision | Recall | F1 |
|------|-----------|--------|-----|
| 句法树方法 | ~60% | ~55% | ~57% |
| 神经网络方法 | ~82% | ~80% | ~81% |

*注：以上为参考结果，实际结果取决于数据集和训练配置*

## 📚 数据集

- **SemCor**: 用于WSD评估，可通过NLTK直接加载
- **PropBank**: 用于SRL评估，需单独下载

详细数据下载说明请参阅 `data/README.md`

## 📖 参考文献

1. Navigli, R. (2009). Word Sense Disambiguation: A Survey. ACM Computing Surveys.
2. Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers.
3. Huang, L., et al. (2019). GlossBERT: BERT for Word Sense Disambiguation.
4. Shi, P., & Lin, J. (2019). Simple BERT Models for Relation Extraction and Semantic Role Labeling.

## 📝 许可证

MIT License
