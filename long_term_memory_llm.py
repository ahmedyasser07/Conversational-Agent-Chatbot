from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
load_dotenv()

model = ChatOpenAI(
    model = "gpt-3.5-turbo",
    temperature = "0.2",
    max_tokens = 1000,
    verbose = True
)

prompt = ChatPromptTemplate.from_messages(
    ("system", "You are a friendly AI assistant"),
    MessagesPlaceholder(variable_name="chat_history")
    ("human", "{input}")
)

memory = ConversationBufferMemory(
    memory_key= "chat_history",
    return_messages= True
)

chain = LLMChain(
    llm=model,
    prompt=prompt,
    memory=memory,
    verbose=True,
)

msg = {
    "input": "hello"
}
response = chain.invoke(msg)