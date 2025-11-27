import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
from supabase.client import Client, create_client

load_dotenv()  


# supabase_url = SUPABASE_URL
# supabase_key = SUPABASE_SERVICE_KEY
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

loader = PyPDFDirectoryLoader("documents")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

BATCH_SIZE = 100

total = len(docs)
print(f"Total chunks to ingest: {total}")

for i in range(0, total, BATCH_SIZE):
    batch = docs[i : i + BATCH_SIZE]
    print(f"Ingesting batch {i}–{i + len(batch) - 1} ...")

    SupabaseVectorStore.from_documents(
        batch,
        embeddings,
        client=supabase,
        table_name="documents",
        query_name="match_documents",
    )

print("✅ Ingestion complete!")

# loader = PyPDFDirectoryLoader("documents")

# documents = loader.load()
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
# docs = text_splitter.split_documents(documents)

# vector_store = SupabaseVectorStore.from_documents(
#     docs,
#     embeddings,
#     client=supabase,
#     table_name="documents",
#     query_name="match_documents",
#     chunk_size=1000,
# )