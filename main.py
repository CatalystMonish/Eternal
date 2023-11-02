from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import SystemMessage, HumanMessage
from langchain.chat_models import ChatOpenAI

# Common declarations for API keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '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')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY', 'db4bd745-b35a-408f-ae24-c242b47ce8e6')
PINECONE_API_ENV = os.getenv('PINECONE_API_ENV', 'gcp-starter')  # You may need to switch with your env

loader = PyPDFLoader("./data/1.pdf")

data = loader.load()

print (f'You have {len(data)} document(s) in your data')
print (f'There are {len(data[30].page_content)} characters in your document')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
texts = text_splitter.split_documents(data)

print (f'Now you have {len(texts)} documents')

# Use common declaration for OpenAI API key
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# initialize pinecone with common declarations for Pinecone API key and environment
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console
)
index_name = "kiran"  # put in the name of your pinecone index here

docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)



llm = ChatOpenAI(
    temperature=0,
    model='ft:gpt-3.5-turbo-0613:personal::8CmXvoV6',
    openai_api_key=OPENAI_API_KEY,
    context=some_context
)

chain = load_qa_chain(llm, chain_type="stuff")

# Get query input from the user
query = input("Enter your query: ")

docs = docsearch.similarity_search(query)
print(docs)

chain.run(input_documents=docs, question=query)
