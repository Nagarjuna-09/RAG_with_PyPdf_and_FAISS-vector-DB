import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter # Can control chunk size while splitting
from langchain_openai import OpenAIEmbeddings, ChatOpenAI, OpenAI
from langchain_community.vectorstores import FAISS

# for retrieval chain
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv()


#************ Loading *********************

pdf_location = ('./documents/React_paper.pdf')

# Creating a pdf loader object
loader = PyPDFLoader(file_path=pdf_location)

# using the load object within the pdf_loader object
# This loader automatically splits the contents in the file and loads it as individual chunks called documents
# But the problem with this splitting is we do not have control of the chunk size if we use
# this automatic splitting. So, we use characterTextSplitter to split these to control the chunk size
documents = loader.load()

#************ Splitting *********************

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
texts = text_splitter.split_documents(documents)

#************ converting split chunks into embeddings using OPENAI *********************

embedding_method = OpenAIEmbeddings()

#************creating vector db and providing documents and embeddingmethod *********************
# Note: You can comment this section once you run it, because after runnning it, an index file is generated storing all the vectors.
# You can directly query from that index file by considering it as vector store instead of generating it everytime you run.

# chunks are converted into vectors and stored on our local machine on our RAM
# By running the below command, the command is run and result is stored in RAM. It is volatile and can be lost anytime.
# If you want to store it to long term, you have to persist it in your harddisk
vectorstore = FAISS.from_documents(documents=documents, embedding= embedding_method)

# saving to local disk for persistent storage.
# we are saving it with the index name faiss_index_react
vectorstore.save_local("faiss_index_react")

# loading from the saved index file and using it as vector store. We are also providing the embedding method to decode it from vectors to text while extracting
# allow_dangerous_deserialization=True is a security measure added by langchain to avoid deserialization attack
# But do not set this to true in production systems
new_vector_store = FAISS.load_local("faiss_index_react",
                                    embedding_method,
                                    allow_dangerous_deserialization=True
                                    )

#***** Writing chain to combine user prompt + retrieved docs/embeddings + instructions *********

# creating retrieval chain
query = "What fine-tuning approach was planned to be explored in conclusion? It starts with H"
llm = ChatOpenAI()
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

retrieval_chain = create_retrieval_chain(
    retriever=new_vector_store.as_retriever(), # vector db to use
    combine_docs_chain=combine_docs_chain # docs to use
)

result = retrieval_chain.invoke(input={"input": query})

print(result)
print(result["answer"])

# Note: This process/code is same as what was done in RAG_wth_Pinecone_Integration project, there we are using Pinecone as vector_store, here we are using FAISS