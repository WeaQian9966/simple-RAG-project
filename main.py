import os
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader

# 设置 API Key (实际项目中建议放在 .env 文件中)
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# os.environ["GOOGLE_API_KEY"] = "asdsa" 

# CHANGED: 引入 Ollama 相关的库
from langchain_community.chat_models import ChatOllama
from langchain_ollama import OllamaEmbeddings

# 1. Loading
# 使用 TextLoader 加载本地文件
# loader = TextLoader("./knowledge.txt", encoding="utf-8")
loader = PyPDFLoader("/Users/wei/Documents/resources/research/1706.03762v7.pdf")
documents = loader.load()
print(f"Loaded {len(documents)} document(s).")


# 2. Splitting
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Chunk size: 每个块大概 500 个字符
# Chunk overlap: 块与块之间重叠 50 个字符，防止上下文在切分点丢失
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)

print(f"Split into {len(chunks)} chunks.")

# from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# 3. Embedding Model
# Gemini
# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# 我们使用本地的 Ollama 来把文本变成向量
# 如果你没下载 nomic-embed-text，这里也可以填 "llama3"
embeddings = OllamaEmbeddings(model="llama3")

# 4. Vector Store
# 创建一个临时的内存数据库，把 chunks 转换成向量存进去
db = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings
)

print("Data embedded and stored in ChromaDB.")

# from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA

# 5. Retrieval & Generation
# 初始化 LLM
# llm = ChatGoogleGenerativeAI(
#     model="gemini-1.5-flash",
#     temperature=0,
#     convert_system_message_to_human=True 
# )

# Ollama
llm = ChatOllama(
    model="llama3", 
    temperature=0, # 0 表示回答更精准，不发散
)

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
# query = "what do i have?"
# response = qa_chain.invoke(query)

# print(f"Q: {query}")
# print(f"A: {response['result']}")

while True:
    query = input("\n请输入问题 (输入 'exit' 退出): ")
    if query == 'exit':
        break
    
    print("思考中...")
    response = qa_chain.invoke(query)
    print(f"Answer: {response['result']}")