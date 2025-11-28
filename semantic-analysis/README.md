# 语义分析项目：词义消歧与语义角色标注

本项目实现了词义消歧(Word Sense Disambiguation, WSD)和语义角色标注(Semantic Role Labeling, SRL)任务，包含多种方法的实现与对比分析。

## 📋 项目简介

### 词义消歧 (WSD)
词义消歧是确定多义词在特定上下文中具体含义的任务。本项目实现了以下方法：

- **Lesk算法**：基于词义定义与上下文词重叠的传统方法
- **BERT上下文方法**：使用预训练模型获取词的上下文嵌入表示
- **知识库增强方法**：结合WordNet图结构和语义关系
- **基于图的方法**：使用PageRank算法在语义图上传播

### 语义角色标注 (SRL)
语义角色标注是识别句子中谓词的论元及其语义角色的任务。本项目实现了：

- **基于句法树的方法**：利用依存句法分析映射语义角色
- **BiLSTM-CRF方法**：基于神经网络的序列标注模型
- **BERT-based方法**：使用BERT进行序列标注

## 🛠️ 环境配置

### 1. 安装依赖
```bash
cd semantic-analysis
pip install -r requirements.txt
```

### 2. 下载必要的模型和数据
```bash
# 下载NLTK数据
python -c "import nltk; nltk.download('wordnet'); nltk.download('semcor'); nltk.download('stopwords'); nltk.download('omw-1.4')"

# 下载spaCy模型
python -m spacy download en_core_web_sm
```

### 3. 安装项目
```bash
pip install -e .
```

## 📁 项目结构

```
semantic-analysis/
├── README.md                    # 项目说明文档
├── requirements.txt             # Python依赖
├── setup.py                     # 安装脚本
├── config/
│   └── config.yaml              # 配置文件
├── data/
│   ├── download_data.py         # 数据下载脚本
│   └── README.md                # 数据说明
├── src/
│   ├── __init__.py
│   ├── wsd/                     # 词义消歧模块
│   │   ├── __init__.py
│   │   ├── base.py              # WSD基类
│   │   ├── context_based.py     # 基于上下文的WSD
│   │   └── knowledge_enhanced.py # 基于知识库增强的WSD
│   ├── srl/                     # 语义角色标注模块
│   │   ├── __init__.py
│   │   ├── base.py              # SRL基类
│   │   ├── syntax_based.py      # 基于句法树的SRL
│   │   └── neural_srl.py        # 基于神经网络的SRL
│   ├── evaluation/              # 评估模块
│   │   ├── __init__.py
│   │   ├── wsd_eval.py          # WSD评估
│   │   └── srl_eval.py          # SRL评估
│   └── utils/
│       ├── __init__.py
│       ├── data_loader.py       # 数据加载器
│       └── preprocessing.py     # 预处理工具
├── experiments/
│   ├── run_wsd.py               # 运行WSD实验
│   ├── run_srl.py               # 运行SRL实验
│   └── compare_methods.py       # 方法对比
└── tests/
    ├── test_wsd.py              # WSD单元测试
    └── test_srl.py              # SRL单元测试
```

## 🚀 使用方法

### 快速演示

#### WSD演示
```python
from src.wsd import LeskWSD

# 创建Lesk算法实例
lesk = LeskWSD()

# 词义消歧
result = lesk.disambiguate(
    context="I went to the bank to deposit money.",
    target_word="bank"
)

print(f"词义: {result.sense_key}")
print(f"定义: {result.definition}")
print(f"置信度: {result.confidence}")
```

#### SRL演示
```python
from src.srl import SyntaxBasedSRL

# 创建SRL实例
srl = SyntaxBasedSRL()

# 语义角色标注
results = srl.predict("The cat ate the fish in the garden.")

for result in results:
    print(f"谓词: {result.predicate}")
    for arg in result.arguments:
        print(f"  [{arg.role}] {arg.text}")
```

### 运行实验

#### 运行WSD实验
```bash
# 运行所有WSD方法
python experiments/run_wsd.py --method all --max_samples 100

# 只运行Lesk算法
python experiments/run_wsd.py --method lesk

# 包含BERT方法（较慢）
python experiments/run_wsd.py --method all --include_bert
```

#### 运行SRL实验
```bash
# 运行基于句法的方法
python experiments/run_srl.py --method syntax

# 运行演示模式
python experiments/run_srl.py --demo
```

#### 方法对比
```bash
# 对比所有方法并生成报告
python experiments/compare_methods.py --task all --report report.md
```

### 运行测试
```bash
# 运行所有测试
pytest tests/ -v

# 运行WSD测试
pytest tests/test_wsd.py -v

# 运行SRL测试
pytest tests/test_srl.py -v
```

## 📊 评估指标

### WSD评估
- **准确率 (Accuracy)**: 正确预测的比例
- **F1分数**: 精确率和召回率的调和平均

### SRL评估
- **精确率 (Precision)**: 正确预测的论元 / 预测的论元总数
- **召回率 (Recall)**: 正确预测的论元 / 正确标注的论元总数
- **F1分数**: 精确率和召回率的调和平均

## 📚 数据集

### WSD数据集
- **SemCor**: 通过NLTK直接加载
- **Senseval/SemEval系列**: 需要单独下载

### SRL数据集
- **PropBank**: 需要单独下载
- **CoNLL-2005/2012**: 需要LDC许可证

详细数据下载说明请参阅 `data/README.md`

## 🔧 配置

编辑 `config/config.yaml` 来修改配置：

```yaml
# 模型配置
models:
  bert:
    model_name: "bert-base-uncased"
    max_length: 512
  
  wsd:
    context_window: 50
    knn_neighbors: 5
```

## 📖 API文档

### WSD模块

#### LeskWSD
```python
class LeskWSD(ContextBasedWSD):
    """Lesk词义消歧算法"""
    
    def disambiguate(self, context: str, target_word: str,
                     target_position: int = None,
                     pos: str = None) -> WSDResult:
        """
        执行词义消歧
        
        Args:
            context: 上下文句子
            target_word: 目标词
            target_position: 目标词位置
            pos: 词性
            
        Returns:
            WSDResult: 消歧结果
        """
```

#### BERTContextWSD
```python
class BERTContextWSD(ContextBasedWSD):
    """基于BERT的上下文词义消歧"""
    
    def __init__(self, model_name: str = "bert-base-uncased"):
        """初始化"""
```

### SRL模块

#### SyntaxBasedSRL
```python
class SyntaxBasedSRL(SRLBase):
    """基于句法树的语义角色标注"""
    
    def predict(self, sentence: str) -> List[SRLResult]:
        """
        对句子进行语义角色标注
        
        Args:
            sentence: 输入句子
            
        Returns:
            SRLResult列表
        """
```

#### BiLSTMCRFSRL
```python
class BiLSTMCRFSRL(NeuralSRL):
    """BiLSTM-CRF语义角色标注模型"""
    
    def train(self, train_data, val_data=None, epochs=10):
        """训练模型"""
    
    def predict(self, sentence: str) -> List[SRLResult]:
        """预测"""
```

## 📝 参考文献

1. Navigli, R. (2009). Word Sense Disambiguation: A Survey. ACM Computing Surveys.
2. Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers.
3. Huang, L., et al. (2019). GlossBERT: BERT for Word Sense Disambiguation with Gloss Knowledge.
4. Shi, P., & Lin, J. (2019). Simple BERT Models for Relation Extraction and Semantic Role Labeling.
5. He, L., et al. (2017). Deep Semantic Role Labeling: What Works and What's Next.

## 📄 许可证

MIT License
