from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from chromadb import Documents, EmbeddingFunction, Embeddings
from tqdm import tqdm  # 引入 tqdm 模块用于显示进度条

class MyEmbeddingFunction(EmbeddingFunction):
     def embed_documents(self, texts: Documents) -> Embeddings:
        # 使用 tqdm 显示嵌入进度条
        embeddings = [
            model.encode(text, convert_to_numpy=True) for text in tqdm(texts, desc="Embedding Texts", unit="text")
        ]
        # 返回嵌入结果
        return [embedding.tolist() for embedding in embeddings]

# Load pre-trained SentenceTransformer model
model = SentenceTransformer('D:/2024 fall/web/lab/实验三/m3e-base')

# Load CSV data
file_path = "law_data_3k.csv"
loader = CSVLoader(file_path=file_path)
data = loader.load()

# Get total record count
total_records = len(data)
print(f"Total number of records: {total_records}")

# Initialize text splitter
text_splitter = CharacterTextSplitter(
    separator="\n",        # Use paragraph separator
    chunk_size=300,         # Set chunk size to 300 characters
    chunk_overlap=100,      # Set overlap between chunks to 100 characters
    length_function=len,    # Use len to calculate text length
    is_separator_regex=False,
)

# Initialize an empty list to hold documents
documents = []

# 记录生成的 chunk 数量
total_chunks = 0

# Process and split text for each record in the CSV
print("Processing records and splitting text...")
for idx, record in enumerate(tqdm(data, desc="Splitting Text", unit="record")):  # 在数据处理部分添加 tqdm 进度条
    text = record.page_content  # Get text content
    
    # Split the document content into chunks
    texts = text_splitter.create_documents([text])
    
    # Add the chunks to the documents list
    documents.extend(texts)

    # 累加生成的 chunk 数量
    total_chunks += len(texts)

 # 打印总共生成了多少个 chunk
print(f"Total number of chunks: {total_chunks}")

# After all records have been processed, create the Chroma database with the embedding function
try:
    # Chroma expects an embedding function, passing MyEmbeddingFunction as the embedding function
    persist_directory = "./chroma_data_300"
    db = Chroma.from_documents(
        documents, 
        MyEmbeddingFunction(),
        persist_directory=persist_directory  
    )
    print("Chroma database saved")
except Exception as e:
    print(f"Error saving Chroma database: {e}")
