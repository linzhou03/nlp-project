# 数据集说明 / Dataset Documentation

## 词义消歧 (WSD) 数据集

### SemCor
- **描述**: 用WordNet词义标注的Brown语料库子集
- **来源**: 通过NLTK直接加载
- **加载方式**:
```python
import nltk
nltk.download('semcor')
from nltk.corpus import semcor
```

### Senseval/SemEval 系列数据集
- **Senseval-2** (2001)
- **Senseval-3** (2004)
- **SemEval-2007 Task 17**
- **SemEval-2013 Task 12**
- **SemEval-2015 Task 13**

这些数据集需要从官方网站下载：
- http://lcl.uniroma1.it/wsdeval/

## 语义角色标注 (SRL) 数据集

### PropBank
- **描述**: 动词论元结构标注
- **网站**: https://propbank.github.io/
- **CoNLL格式**: 使用CoNLL-2005/2012格式

### CoNLL-2005 Shared Task
- **描述**: 语义角色标注共享任务数据
- **需要LDC许可证**: 包含Wall Street Journal数据

### CoNLL-2012 Shared Task
- **描述**: OntoNotes 5.0数据集
- **网站**: http://conll.cemantix.org/2012/

## 数据下载

运行以下命令下载可用数据：
```bash
python download_data.py --all
```

或者分别下载：
```bash
# 下载WSD数据
python download_data.py --wsd

# 下载SRL数据  
python download_data.py --srl
```

## 数据目录结构

```
data/
├── raw/                    # 原始数据
│   ├── wsd/               # WSD数据
│   │   ├── semcor/
│   │   ├── senseval2/
│   │   └── semeval/
│   └── srl/               # SRL数据
│       ├── propbank/
│       └── conll/
├── processed/             # 处理后的数据
│   ├── wsd/
│   └── srl/
└── cache/                 # 缓存数据
```

## 数据格式

### WSD数据格式
```json
{
    "sentence": "I went to the bank to deposit money.",
    "target_word": "bank",
    "target_position": 4,
    "sense_key": "bank%1:14:00::",
    "lemma": "bank"
}
```

### SRL数据格式 (CoNLL)
```
1    The    DT    B-ARG1
2    cat    NN    I-ARG1
3    ate    VBD   B-V
4    fish   NN    B-ARG2
5    .      .     O
```

## 注意事项

1. 部分数据集需要学术许可证
2. 下载大型数据集时请确保网络稳定
3. 处理后的数据会缓存以加速后续加载
