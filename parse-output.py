from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser, JsonOutputParser
from  langchain_core.pydantic_v1 import  BaseModel, Field


load_dotenv()

llm = ChatOpenAI(
    model = "gpt-3.5-turbo",
    temperature = "0.2",
    max_tokens = 1000,
    verbose = True
)

def parse_output_str(input):

    prompt = ChatPromptTemplate.from_messages(
        ("system", "You are an English teacher, write a 3 line poem including the following words"),
        ("human", "{input}")
    )

    parser = StrOutputParser() #parses AI response message to string

    chain = prompt | llm | parser

    response = chain.invoke({"input": input})
    return response


def parse_output_csv_list(input):

    prompt = ChatPromptTemplate.from_messages(
        ("system", "You are an English teacher, generate a list of 10 synonyms, return the result as comma seperated values"),
        ("human", "{input}")
    )

    parser = CommaSeparatedListOutputParser() #parses AI response CSV message to list of strings

    chain = prompt | llm | parser

    response = chain.invoke({"input": input})
    return response


def parse_output_JSON(input):

    prompt = ChatPromptTemplate.from_messages(
        ("system", "Extract information from the following phrase.\n Formatting instructions: {format_instructions} "),
        ("human", "{input}")
    )

    class Person(BaseModel):
        name: str = Field(description= "name of the person")
        age: int = Field(description= "age of the person")
        nationality: str = Field("nationality of the person")

    parser = JsonOutputParser(pydantic_object = Person) #parses AI response message to JSON based on the instructions
                                                        # which in our case is the person's fields

    chain = prompt | llm | parser

    response = chain.invoke(
        {
            "input": input,
            "format_instructions": parser.get_format_instructions()
        }
    )
    return response


print(parse_output_str("love"))
print(parse_output_csv_list("happiness"))
print(parse_output_JSON("ahmed is 19 years old and he is Egyptian"))
