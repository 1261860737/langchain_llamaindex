# langchain_llamaindex

该智能体为一个多模型智能对话与知识库系统，支持多种大模型平台（Ollama、vLLM、Qwen、OpenAI、智谱AI等），集成基础对话、RAG增强对话、Agent智能体对话三大模式，并内置丰富的知识库检索与外部工具调用能力。

RAG：采用llamaindex框架，支持bge-m3等多种embedding模型，集成FlashRank、Cohere、Jina等多种重排序模型，提升检索准确率和文档排序质量，并通过Few-shot Prompting增强复杂问题的泛化能力。

tools：内置DuckDuckGo、SerpAPI、Wikipedia、ArXiv、每日AI论文、天气查询等多种实用工具。

---

---
## 安装与运行
### 1. 环境准备
- Python >= 3.9（推荐3.10）
- 建议使用conda：
```bash
conda create -n kk-lite python=3.10 -y
conda activate kk-lite
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 启动本地模型（可选）
- Ollama：请参考 [Ollama官方文档](https://ollama.com/) 部署

- vLLM：下载模型至 ./model 目录，启动服务：
```bash
vllm serve --model /path/to/model \ 
   --tensor-parallel-size 6 \            # 并行数=GPU数量
   --gpu-memory-utilization 0.85 \       # 显存利用率阈值（6卡建议0.8~0.9）
   --max-num-seqs 64 \        # 高并发优化
   --enforce-eager \          # 避免多卡兼容问题
   --port 8011 \              # 服务端口
   --api-key "your-token"     # 访问令牌（增强安全性）
```
### 4.设置env
```bash
cp .env.example .env
```
在.env中填写自己的Api-key和Base-url

### 5. 启动Web界面
```bash
streamlit run streamlit_main.py --theme.primaryColor "#165dff"
# 或暗色模式：
streamlit run streamlit_main.py --theme.base "dark" --theme.primaryColor "#165dff"
```







