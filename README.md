# L-IA-dans-l-agriculture

该农业软件使用了两个AI模型
1、wisper
2、llama3-1b
## 第一个模型：
使用 Hugging Face Transformers 与 PyTorch 微调 Whisper 模型，结合 whisper.cpp 实现高效英文语音识别系统，覆盖数据采集、模型训练与部署全流程，优化农业领域词汇识别效果。

## 第二个模型：
由于设备所限，使用llama3-1b的模型，再google-colab上运行微调模型，并且只微调了LoRA层，实现了llama3+RAG（检索增强生成）  
RAG:使用了FAISS向量数据库，支持txt,json，~~pdf~~格式的数据  

### 使用
使用请运行 **google_colab_test.py**  
第一次使用需要运行**data_processor.py**创建FAISS向量数据库  
如需添加txt、json格式的数据请自行修改相关功能文件

### 评估
请运行**run_evaluation.py**  

### 最终检测结果：  
✅ BERTScore: 0.3385  
✅ Perplexity (PPL): 13.7480  
符合预期  

PS:由于数据收集、处理太麻烦只对转成FAISS向量的数据进行了简单的处理，并且有处理pdf的功能但是过于简单，如有需求自行修改
