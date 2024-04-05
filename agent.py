from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain.tools.retriever import create_retriever_tool

from dotenv import load_dotenv
load_dotenv()

#retrievar
url = "https://www.formula1.com"
loader = WebBaseLoader(url)
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)
splitDocs = splitter.split_documents(docs)
embedding = OpenAIEmbeddings()
vector_store = FAISS.from_embeddings(docs, embedding=embedding)
retriever = vector_store.as_retriever(search_kwargs = {"k":3})




llm = ChatOpenAI(
    model = "gpt-3.5-turbo",
    temperature = "0.7",
)

prompt = ChatPromptTemplate.from_messages(
    ("system", "You are friendly assistant called Max"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
)

search = TavilySearchResults()
retriever_tool = create_retriever_tool(
    retriever,
    "formula1_search",
    "Use this tool when searching for information about Formula1 (F1)."
)
tools = [search, retriever_tool]

agent = create_openai_functions_agent(
    llm= llm,
    prompt=prompt,
    tools=tools
)

agentExecutor = AgentExecutor(
    agent=agent,
    tools=tools

)
def create_chatbot_response(agentExecutor, user_input, chat_history):

    response = agentExecutor.invoke({
        "input": user_input,
        "chat_history": chat_history
    })
    return response["output"]


if __name__ == " __main__":


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
        response = create_chatbot_response(agentExecutor, user_input, chat_history)

        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))

        print("Chatbot Response: ", response)

    print("Thank You for using the chatbot!")