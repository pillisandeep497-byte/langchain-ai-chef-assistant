from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
from pydantic import BaseModel,Field
import os 
load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

class Cheif(BaseModel):
    ingrediants:str = Field(description="give detailed ingrediants of input food item")
    precautions:str = Field(description="give precautions during cooking")
    process:str = Field(description="give exact detailed process of making food ")
    optional:str = Field(description="give some optional steps to cook")


parser = PydanticOutputParser(
    pydantic_object=Cheif
)

promt = ChatPromptTemplate.from_messages([
    "system",
    """
1.impliment art rule for output:
    A means act as a senior cheif
    R means request 
    T means terms 
      
2.give output in required format.
{format_instructions} """,
MessagesPlaceholder("history"),
("human"),("{input}")]
)

chain = promt.partial(
    format_instructions=parser.get_format_instructions
)|llm|parser

store = {}
def get_history(session_id:str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()

    return store[session_id]

mentor = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history"

)

print("==================================AI senior cheif===============================")
print("type exit to quit")

while True:
    user_input=input("you: ")
    if user_input.lower() == "exit":
        break

    response = mentor.invoke(
        {"input":user_input},
        config={
            "configurable":{"session_id":"sandeep"}
        }
    )
    print("\ningrediants: ",response.ingrediants)
    print("\nprecautions: ",response.precautions)
    print("\nprocess: ",response.process)
    print("\noptional: ",response.optional)