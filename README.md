# simple-RAG-project

**Retrieval-Augmented Generation (RAG)** 是目前将 **LLM** (Large Language Model) 与私有数据结合最主流的  **Architecture** 。

它解决了 LLM 的两大痛点：

1. **Hallucination** (幻觉)：胡说八道。
2. **Knowledge Cutoff** (知识截止)：不知道最新的或私有的信息。

我们把这个项目分为三个阶段：**Concept** (概念)、**Stack** (技术栈)、**Implementation** (代码实现)。

---

### 第一阶段：理解核心流程 (The Workflow)

RAG 的本质其实就是“开卷考试”。当用户提问时，系统先去翻书（你的数据），找到相关的段落，然后把问题和段落一起交给 LLM，让它根据段落回答。

流程通常分为五个步骤：

1. **Loading** : 加载你的私有数据（PDF, Wiki, Txt）。
2. **Splitting (Chunking)** : 把长文章切成小的 **Chunks** (片段)。
3. **Embedding** : 把文本转化成计算机能理解的 **Vector** (向量/数字列表)。
4. **Storing** : 把向量存入 **Vector Database** (向量数据库)。
5. **Retrieval & Generation** : 用户提问 -> 搜索相似片段 -> 组装 **Prompt** -> LLM 生成回答。

---

### 第二阶段：选择技术栈 (Tech Stack)

对于初学者（From Scratch），我推荐使用 Python 和最标准的“黄金组合”，既简单又强大：

* **Orchestration Framework** (编排框架): **LangChain** (最流行) 或 **LlamaIndex** (专注数据索引)。我们今天用 LangChain。
* **LLM Provider** : **OpenAI (GPT-3.5/4)** (最稳) 或 **Ollama** (本地运行 Llama 3，免费且隐私)。为了演示方便，我们假设使用 OpenAI，如果你想用本地模型，只需改一行代码。
* **Vector Database** : **ChromaDB** 或  **FAISS** 。这两个都可以在本地运行，不需要复杂的服务器配置。
* **Embedding Model** : OpenAI `text-embedding-3-small` 或 HuggingFace 的免费模型。

---

### 第三阶段：一步步代码实现 (Implementation)

首先，你需要安装必要的 **Dependencies** (依赖库)。打开你的  **Terminal** ：

**Bash**

```
pip install langchain langchain-openai chromadb python-dotenv
```

创建一个 `main.py` 文件，我们开始写代码。

#### Step 1: Data Loading (加载数据)

假设你有一个文本文件 `knowledge.txt`，里面写着你公司的私有信息（比如：“我喜欢吃麻辣火锅”）。

**Python**

```
import os
from langchain_community.document_loaders import TextLoader

# 设置 API Key (实际项目中建议放在 .env 文件中)
os.environ["OPENAI_API_KEY"] = "sk-..." 

# 1. Loading
# 使用 TextLoader 加载本地文件
loader = TextLoader("./knowledge.txt", encoding="utf-8")
documents = loader.load()

print(f"Loaded {len(documents)} document(s).")
```

#### Step 2: Splitting / Chunking (文本分块)

LLM 的 **Context Window** (上下文窗口) 是有限的，而且把整本书塞进去太贵且不精准。我们需要把文本切成小块。

**Python**

```
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 2. Splitting
# Chunk size: 每个块大概 500 个字符
# Chunk overlap: 块与块之间重叠 50 个字符，防止上下文在切分点丢失
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)

print(f"Split into {len(chunks)} chunks.")
```

#### Step 3 & 4: Embedding & Vector Store (向量化与存储)

这是 RAG 的魔法所在。我们将文本转化为  **High-dimensional Vectors** ，并存入数据库。

**Python**

```
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# 3. Embedding Model
embeddings = OpenAIEmbeddings()

# 4. Vector Store
# 创建一个临时的内存数据库，把 chunks 转换成向量存进去
db = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings
)

print("Data embedded and stored in ChromaDB.")
```

#### Step 5: Retrieval & Generation (检索与生成)

现在数据库准备好了，我们来构建 **Chain** (链)。

**Python**

```
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# 5. Retrieval & Generation
# 初始化 LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# 创建检索器 (Retriever)
# k=2 意思 是只找最相似的 2 个片段
retriever = db.as_retriever(search_kwargs={"k": 2})

# 构建问答链 (QA Chain)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", # "stuff" 意味着把找到的片段直接塞进 Prompt
    retriever=retriever
)

# --- 测试一下 ---
query = "我喜欢吃什么？"
response = qa_chain.invoke(query)

print(f"Q: {query}")
print(f"A: {response['result']}")
```

---

### 第四阶段：关键概念解析 (Key Concepts Review)

在运行上述代码时，有几个核心术语你必须掌握：

1. Semantic Search (语义搜索):
   传统数据库用关键词匹配（Ctrl+F），Vector Database 用语义匹配。
   * *Example:* 你搜 "Dog"，它能找到 "Puppy" 或 "Canine"，因为它们在向量空间里距离很近。
   * 数学原理通常基于 **Cosine Similarity** (余弦相似度)。
2. Prompt Engineering (提示工程):
   LangChain 在后台默默做了一个 Prompt 模板，大概长这样：
   > "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know..."
   >
   > [这里插入检索到的 Chunks]
   >
   > Question: [你的问题]
   >
3. Ingestion Pipeline (摄取管道):
   Step 1 到 Step 4 通常被称为 Ingestion。在实际生产环境中，这是一次性的或者定时的任务，而不是每次提问都跑一遍。

---

### 下一步建议 (Next Steps)

你现在已经跑通了一个最小可行性版本 (MVP)。想要进阶，你可以尝试：

1. **Replace OpenAI** : 尝试用 Ollama 替换 OpenAI，实现完全免费的本地 RAG。
2. **Different Data Source** : 尝试加载 PDF (`PyPDFLoader`) 或者 网页 (`WebBaseLoader`)。
3. **UI Interface** : 使用 **Streamlit** 给你的代码加一个简单的网页界面。
