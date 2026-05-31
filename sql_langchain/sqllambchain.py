from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from sqlalchemy import create_engine

# Connect to a business database
engine = create_engine("sqlite:///business_data.db")
db = SQLDatabase(engine)

# Set up LLM-powered SQL chain
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
chain = create_sql_query_chain(llm, db)

# Query the database using natural language
question = "What is the total revenue this month?"
sql_query = chain.invoke({"question": question})
response = db.run(sql_query)
print(response)
