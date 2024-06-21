# The following streamlit project contains two modules
# 1) knowledge graph construction from free text using LLM Graph Transformers.
# 2) Chat with the constructed knowledge graph using the LLM graph cypher chain
# Note: This is example and there are ways where we can improve both the knowledge graph construction and communication modules
##################################################################################################################################
 
import streamlit as st
from streamlit_option_menu import option_menu
import os
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)


# 1. as sidebar menu
with st.sidebar:
    side_menu = option_menu("Main Menu", ["Home", 'Settings'], 
        icons=['house', 'gear'], menu_icon="cast", default_index=1)
    side_menu

# 3. CSS style definitions
main_menu = option_menu(None, ["Home", "Gchat",  "Tasks", 'Settings'], 
    icons=['house', 'cloud-upload', "list-task", 'gear'], 
    menu_icon="cast", default_index=0, orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "green"},
    }
)

st.session_state['main_menu'] = main_menu

####################### MENU ####
#graph = Neo4jGraph()
# Connect the running neo4j graph instance with LLM langchain
graph = Neo4jGraph('bolt://localhost:7687','neo4j','abcd1234')

try:
    graph.connect()
    print("Connected to Neo4j instance successfully!")
except Exception as e:
    print("Failed to connect to Neo4j instance:", e)

#llm = ChatOpenAI(temperature=0, model_name="gpt-4-turbo")
#llm_transformer = LLMGraphTransformer(llm=llm)

#*****************************************************************
# OpenAI
#*****************************************************************
# Only required when using OpenAI LLM or embedding model

OPENAI_API_KEY="user-open-api-key"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


#llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
llm = ChatOpenAI(temperature=0, model_name="gpt-4o-2024-05-13")

################### prompt engineering ########################
# examples are used to improve responses from Knowledge graph
# these are used as template and we can add different examples
###############################################################

examples = [
    {
        "question": "who plays in al hilal?",
        "query": "MATCH p=()-[r:PLAYS_FOR]->(Club{{id:'Al Hilal'}}) RETURN p"
    },
    {
        "question": "who plays for al hilal?",
        "query": "MATCH p=()-[r:PLAYS_FOR]->(Club{{id:'Al Hilal'}}) RETURN p"
    },
    {
        "question": "who is Mohamed Salah?",
        "query": "MATCH p=(Person{{id:'Mohamed Salah'}})-[r:NATIONALITY]->() RETURN p"
    },
    {
        "question": "which club Mohamed Salah plays for?",
        "query": "MATCH p=(Person{{id:'Mohamed Salah'}})-[r:PLAYS_FOR]->() RETURN p"
    },
    {
        "question": "who plays in forward position?",
        "query": "MATCH p=()-[r:POSITION]->(Position{{id:'Forward'}}) RETURN p"
    },
    {
        "question": "who plays in Defender position?",
        "query": "MATCH p=()-[r:POSITION]->(Position{{id:'Defender'}}) RETURN p"
    }
]


example_prompt = PromptTemplate.from_template(
    "User input: {question}\nCypher query: {query}"
)
prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="You are a Neo4j expert. Given an input question, create a syntactically correct Cypher query to run.\n\nHere is the schema information\n{schema}.\n\nBelow are a number of examples of questions and their corresponding Cypher queries.",
    suffix="User input: {question}\nCypher query: ",
    input_variables=["question", "schema"],
)

#st.warning(prompt.format(question="who plays in forward position?", schema=graph.schema))

################### prompt engineering ####################

def main():

    if st.session_state['main_menu'] == 'Home':
        # construct graph-transformer object
        llm_transformer = LLMGraphTransformer(llm=llm)
        st.title("LLM Graph Generation")
        
        # we can construct graphs from PDF docs, and text files but in this example we use a text area
        # Text area for user input
        user_input = st.text_area("Enter some text:")
        
        # Button to process the text
        if st.button("Process"):
            # in the process function we can add text pre-processing techniques if necessary.
            processed_text = process_text(user_input)
            
            # create knoweldge graph nodes and edges from the user provided text.
            st.info('creating graph')
            documents = [Document(page_content=user_input)]
            graph_documents = llm_transformer.convert_to_graph_documents(documents)
            
            # we need a Neo4j graph instance up and running and the following code will populate the extracted nodes and edges to the running neo4j graph.
            st.warning('Populating Neo4j DB')
            graph.add_graph_documents(graph_documents)

            # Referesh graph schema after updates so that LLM knows the updates. we run this each time we add new nodes and edges to the graph it will evolve automatically.
            graph.refresh_schema()

            #print(f"Nodes:{graph_documents[0].nodes}")
            #print(f"Relationships:{graph_documents[0].relationships}")

            st.warning('knowledge graph construction done.')
            ## Here are nodes:
            st.write(graph_documents[0].nodes)

            ## Here are relationships
            st.info(graph_documents[0].relationships)
            
    if st.session_state['main_menu'] == 'Gchat':
        # Below we will chatt with the knowldge graph using the LLM graph cypher chain
        st.title("LLM Graph Chat")
        user_question = st.text_area("Ask Question:")
        user_question = process_text(user_question)

        chain = GraphCypherQAChain.from_llm(graph=graph, llm=llm, cypher_prompt=prompt, verbose=True)
        if st.button('Ask'):
            response = chain.invoke({"query": user_question})
            st.write(response)


def process_text(text):
    # Just a simple processing function for demonstration
    return text.lower()

if __name__ == "__main__":
    main()