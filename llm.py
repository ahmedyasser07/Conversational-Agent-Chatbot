from dotenv import load_dotenv #import the environment variables from .env

from langchain_openai import ChatOpenAI #import openAI package from LangChain
from langchain_core.prompts import ChatPromptTemplate



load_dotenv() #load environment variables

llm = ChatOpenAI( #instantiate openAI model with API_Key from .env
    model = "gpt-3.5-turbo", #choose model
    temperature = "0.2", #choose temperate (how creative the responses are (0 to 1, 1 being most creative)
    max_tokens = 1000, #number of tokens per response (use to limit the response size and control expenses
    verbose = True #debug the response if true
)



#$$$ Simple methods to quickly interact
#llm.invoke -> ask 1 question
#llm.batch -> ask multiple questions in parallel
#llm.stream() -> ask a question and recive the response in chunks (ex: every word is a chunk)


#$$$ Chaining

#create a prompt that will ask a question and make a parameter (later) to make the question dynamic
prompt = ChatPromptTemplate.from_template("what is a {subject}")

#create a prompt that makes the model more specific about a topic
pt = ChatPromptTemplate.from_messages(
    ("system", "You are an English teacher, write a 3 line poem including the following words"), #specify the role of the AI model
    ("human", "{input}") #dynamic inputs from user as parameter
)




# chaining here works like-> input given to prompt, output of prompt goes as input to llm,
# and the output of the llm, and so on for more chains
chain = prompt | llm

chain2 = pt | llm

#here we pass a JSON object or a dictionary object specifying the parameter of the prompt above which was ex: {subject}
responses = chain.invoke({"subject": "car"})
response2 = chain2.invoke(({"input": "love"}))



