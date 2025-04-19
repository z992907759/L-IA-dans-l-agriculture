# 🌱 L-IA-dans-l'agriculture

一个结合语音识别与大语言模型的智能农业 AI 系统，涵盖语音转文字、语义检索、智能问答、网页展示等完整流程。

---

## 🧠 使用的 AI 模型

### 1. Whisper（语音识别）

- 使用 Hugging Face Transformers 与 PyTorch 微调 Whisper 模型；
- 融合 `whisper.cpp` 实现高效、轻量的英文语音识别；
- 支持农业领域专业词汇识别；
- 数据来源于 Mozilla Common Voice，并经过自定义增强。

**使用方式：**

- 请在 `master` 分支中下载已训练好的 Whisper 模型；
- 本地运行即可调用识别功能。

---

### 2. LLaMA 3 + LoRA + RAG（智能问答）

- 使用 LLaMA3-1B（Meta 发布的轻量大语言模型）；
- 在 Google Colab 上进行微调，仅训练 LoRA 层，节省显存；
- 实现 RAG（Retrieval-Augmented Generation）能力；
- 使用 FAISS 构建知识向量库，支持 `.txt`、`.json` 数据格式。

**使用方式：**

1. 第一次运行请先执行：

   ```bash
   python data_processor.py
   ```

   创建 FAISS 知识向量库。

2. 启动后端服务：

   - App 端：运行 `google_colab_test.py`；
   - 网页端：运行 `Run.py`。

3. 若需添加知识数据：

   - 将 `.txt` 或 `.json` 文件放入：

     ```
     Llama3-1b/database/knowledge_base
     ```

   - 然后重新运行 `data_processor.py`。

**评估方式：**

```bash
python run_evaluation.py
```

**模型效果：**

- ✅ BERTScore: `0.3385`
- ✅ Perplexity (PPL): `13.7480`

> ⚠️ 当前支持 `.txt` 与 `.json`，`.pdf` 的处理功能初步实现，效果有限。如需增强请修改 `data_processor.py`。

---

## 💻 前端启动说明（React + TypeScript）

⚠️ 本项目未上传 `node_modules` 目录，请手动安装依赖！

### 第一步：进入前端目录

```bash
cd frontend
```

### 第二步：安装依赖

```bash
npm install
# 或
# yarn install
```

### 第三步：启动开发服务器

```bash
npm start
# 或
# yarn start
```

启动成功后，请访问：[http://localhost:3000](http://localhost:3000)

---

### 注意事项：

- 必须先安装 Node.js（建议版本 ≥16）；
- 只需保留 `package.json` 和 `package-lock.json` 即可；
- 每次 clone 项目后，运行 `npm install` 会自动还原依赖；
- 不建议上传 `node_modules`，它是自动生成的依赖目录。

---

## 🧩 后端服务启动说明（Spring Boot）

项目后端基于 Spring Boot 构建，提供 REST 接口，供前端与 AI 模型进行通信。

### 启动方式：

#### ✅ 方法一：使用 IDE（推荐）
在 IntelliJ IDEA 或 VS Code 中：

1. 打开项目；
2. 找到 `AiChatApplication.java`；
3. 右键点击 `main()` 方法 → 选择“Run”。

#### ✅ 方法二：使用命令行（终端）

确保你已经安装了 Maven 或使用项目自带的 Wrapper：

```bash
# 使用内置 Maven Wrapper 启动
./mvnw spring-boot:run

# 或者使用本地 Maven 启动
mvn spring-boot:run
```

启动成功后，后端服务默认运行在：
```
http://localhost:8080
```

---

## 📁 项目结构简览

```
L-IA-dans-l-agriculture/
├── whisper/                 # Whisper 模型及训练脚本
├── Llama3-1b/               # LLaMA3 微调及问答系统
│   ├── database/            # 知识数据
│   ├── data_processor.py    # 向量数据库构建脚本
│   ├── run_evaluation.py    # 模型评估脚本
│   ├── Run.py               # Web 服务后端入口
├── frontend/                # 网页端 React 项目
│   ├── package.json         # 项目依赖列表
│   └── src/                 # React 页面组件
├── ai-chat-backend/         # Spring Boot 后端服务
│   └── src/com/aichat/AiChatApplication.java
```

---

## 🧪 技术栈

- **AI 模型**：Whisper（语音识别），LLaMA3 + LoRA + RAG（问答）
- **后端**：Java + Spring Boot + RESTful API
- **前端**：React + TypeScript + Axios + Material UI
- **模型训练**：Google Colab（PyTorch + PEFT + FAISS）
- **数据库**：FAISS 向量库（本地检索增强）

---

## 🙏 致谢

感谢 Mozilla、Meta、Hugging Face 等开源社区提供的模型与工具。
