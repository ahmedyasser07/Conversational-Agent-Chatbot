from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain

load_dotenv()

def get_documents_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    splitDocs = splitter.split_documents(docs)
    print(len(splitDocs))
    return splitDocs

def create_vector_db(docs):
    embedding = OpenAIEmbeddings()
    vector_store = FAISS.from_embeddings(docs, embedding=embedding)
    return  vector_store

def create_chain(vector_store):
    model = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature="0.4",
    )

    prompt = ChatPromptTemplate.from_template(
        "Answer the user's question. "
        "Context: {context}"
        "Question: {question}"
    )

    # chain = prompt | model
    chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt
    )

    retriever = vector_store.as_retriever()

    retrieval_chain = create_retrieval_chain(
        retriever,
        chain
    )
    return retrieval_chain


docs = get_documents_from_web("https://www.formula1.com")

vector_store = create_vector_db(docs)

chain = create_chain(vector_store)



result =chain.invoke(
    {
        "question": "what is a F1",
    }
)

print(result["answer"])