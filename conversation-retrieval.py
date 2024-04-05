from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.history_aware_retriever import create_history_aware_retriever
load_dotenv()

def get_documents_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=20
    )
    splitDocs = splitter.split_documents(docs)
    print(len(splitDocs))
    return splitDocs

def create_vector_db(docs):
    embedding = OpenAIEmbeddings()
    vector_store = FAISS.from_embeddings(docs, embedding=embedding)
    return  vector_store

def create_chain(vectore_store):
    model = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature="0.4",
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "answer the following questions based on the following context: {context}."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),

    ])

    # chain = prompt | model
    chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt
    )

    retriever = vectore_store.as_retriever(search_kwargs= {"k":3} )

    retriever_prompt=ChatPromptTemplate.from_messages(
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("human", "Given the above conversation, generate a search query to look up in order to get relevant information to the conversation")

    )

    history_aware_retriever = create_history_aware_retriever(
        llm= model,
        retriever=retriever,
        prompt= retriever_prompt
    )

    retrieval_chain = create_retrieval_chain(
        history_aware_retriever,
        chain
    )
    return retrieval_chain


def create_chatbot_response(chain, question, chat_history):
    result = chain.invoke({
        "question": question,
        "chat_history": chat_history
    })
    return result["answer"]





if __name__ == " __main__":
    docs = get_documents_from_web("https://www.formula1.com")
    vector_store = create_vector_db(docs)
    chain = create_chain(vector_store)

    chat_history = [
        HumanMessage(content="hello"),
        AIMessage(content="Hello, how may I assist you today?"),
        HumanMessage(content="My name is Ahmed Abdelfatah")
    ]

    print('Hello to Chatbot, you can use this as much as you want, whenever you are done type "exit".')

    while True:
        user_input = input("Your Question: ")
        if user_input.lower() == 'exit':
            break
        response = create_chatbot_response(chain, user_input, chat_history)

        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))

        print("Chatbot Response: ", response)

    print("Thank You for using the chatbot!")