import asyncio
import logging
import os
from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict
import sys

import chainlit as cl
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.input_widget import Select, Switch, Slider
from langchain_tavily import TavilySearch
from chainlit.types import ThreadDict
from fastapi import Request, Response
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessageChunk, ToolMessage
from langchain.schema.runnable.config import RunnableConfig
from langchain_openai import ChatOpenAI
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row

#the graph is a blueprint built the includes the chosen settings
# but only state/history is persisted by the checkpointer during runs.



#this is for windows users since it assumes linux
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

#initializing tool, i do max_results = 2 which takes the top 2 most relevant results from these sources
tavily_search = TavilySearch(
    max_results=2,
    topic="general",
    include_domains=["clevelandclinic.org", "mayoclinic.org", "cdc.gov", "nih.gov", 
                    "medlineplus.gov", "medlineplus.gov", "webmd.com", "wikipedia.org"]
)

#initializing db connection pool as none for now
db_connection_pool: Optional[AsyncConnectionPool] = None
#im using the gpt 4o mini model, but you can change it to whatever one you want
models = ["gpt-4o-mini"]
#putting the tavily tool we initialized earlier in line 29 in our list of tools. this list will be used later
tools = [tavily_search]
system_prompt = ("You are a helpful health assistant. "
            "Use the Tavily Search tool to collect information to answer the user's health related questions")
#you can change the prompt to whatever you want, but this string will be passed in later
logger = logging.getLogger(__name__)

def create_graph(model: str,  temperature: float, tools: list, system_prompt: str) -> CompiledStateGraph:

#this is from my environment file
    base_url = os.getenv("BASE_URL")

#initializing 
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        base_url=base_url,
        streaming=True)
    #our chatbot node uses these settings, so altho the settings dont get a node, they are reflected thru in the graph
    #not saved tho bc only our state (messages only) is saved in the db on the message sending checkpoints
    
    #llm_with_tools is gpt-4o but we give it the tavily search tool we initialized earlier
    llm_with_tools = llm.bind_tools(tools)


#in langgraph is it built in all the nodes see the state aka here, the message history per chat
#each chat can only see thier own history/state
    class State(TypedDict):
        messages: Annotated[list, add_messages]
    # our schema only has messages, in format of add_messages
#stores as type:humann, message .. in db

    graph_builder = StateGraph(State)

    async def chatbot(state: State):
        return {"messages": [await llm_with_tools.ainvoke([SystemMessage(content=system_prompt)] + state["messages"])]}

    def should_continue(state: State) -> Literal["tools", "__end__"]:
        messages = state["messages"]
        last_message = messages[-1]
# we need this part for a few lines below. this is how we determine if we need to call tools.
#you need tools when someone asks something that the model itself doesnt know and needs to do a web search to get info from an api
        if last_message.tool_calls:
            return "tools"
        #we return the string tools if we need to call tools as a marker
        
        return "__end__"
    #end means no tools needed to be called
    #the two cases ^

    tool_node = ToolNode(tools)
    #type of node, we put our tools list in it
#langgraph stuff
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", tool_node)
    #our two nodes

    graph_builder.add_edge(START, "chatbot")
    #the start edge comes from chatbot node, aka, it always starts at the chatbot node
    graph_builder.add_conditional_edges(
        "chatbot",
        should_continue,
        #we use should_continue() as a callback function, its itll call it when needed
        {"tools": "tools", "__end__": END}
        #if the llm asked for tools → go to "tools", if not → go to END
    )
    graph_builder.add_edge("tools", "chatbot")
    #edge between tools and chatbot gets added if we had to call tools, because end just END, doesnt reach here

    return graph_builder
#returning our graph


@cl.on_app_startup
async def on_app_startup():
    #we make it global because we need it to be accessed via the entire program for processes to connect to the database to store data
    global db_connection_pool
    #making a connection pool for the langgraph data base
    # #diff tasks will connect to this to store data so we can have persistent data

    #here, we say that a minimum of one task can BORROW A CONNECTION, and at most 8. you can change this if you want when running it
    #once the task is done with the connection (ex you switch chats) it returns it back to pool
    #keep in mind only one agent can be alive at once, one per chat
    db_connection_pool = AsyncConnectionPool(os.getenv("LANGGRAPH_DB_URL"), open=False,
                                            min_size=1, max_size=8,
                                            kwargs={
                                                        "autocommit": True, #every SQL statement is commited automatically to db
                                                        "row_factory": dict_row,
                                                    }
                                            )
    
    await db_connection_pool.open(wait=True)
    #thread can go do other tasks as it waits for this
    #here upon the event we start up the app, we are initialzing our connection pool needed for the rest of the program
    logger.info("App startup: initialized database connection pool")

@cl.data_layer
def get_data_layer():
    return SQLAlchemyDataLayer(conninfo=os.getenv("DATABASE_URL"))
#we get the database_irl from our env file, itll be diff for whoever runs this application bc it depends on ur computer/setup


#keep in mind: this "@cl.()" is a decorator, we use it to wrap around the function we write
#the function decorating aka the thig next to @ is the function we are using as a wrapper, its a function that is alerted when the event in its name occurs
#then we use that to wrap what we write so that way when the event is sensed (from built in chainlit function), our code runs
#(basically we are just writing event handlers, it kind of registers the stuff we write with built in chainlit without modifying either)
@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
    """Password auth handler for login"""
    
    # for development only, DO NOT use in production.
    # fecure authentication and authorization methods are necessary for production deployment
    #here we dont care about passwords, so you can enter whatever password as long as user is admin
    if username == "admin": 
        return cl.User(identifier="admin", metadata={"role": "ADMIN"})
    else: 
        return None

@cl.on_chat_start
async def on_chat_start():
    user = cl.user_session.get("user")
    chainlit_thread_id = cl.context.session.thread_id
    logger.info(f"{user.identifier} started a new chat with thread id {chainlit_thread_id}")
    #this gets printed to terminal for our debugging purposes

#here we set up the settings menu
    settings = await cl.ChatSettings(
        [        
            #chainlit has built in Select and Slider objects for ChatSettings    
            Select(
                id="LLM", #whatever you select in dropdwon is stored in "LLM"
                label="OpenAI model to use",
                values=models,
                initial_value=models[0]
                #we only have one model in our model list (gpt 4o) so this is the only thing on dropdown
            ),

            Slider(
                id="Temperature",
                label="Temperature of the LLM",
                initial=0,
                min=0,
                max=1,
                step=0.1
                #a slider for temperature (aka randomness/creatvity. for the purposes of this bot, i recommend making it low because these are health issues)
            )
        ]
    ).send()

    cl.user_session.set("graph_builder", 
                        create_graph(
                                model=settings['LLM'], #here we take what we stored in the ids annd save it in a graph
                                temperature=settings["Temperature"],
                                tools=tools,
                                system_prompt=system_prompt))
    #this graph is for storing settings
    
    await cl.Message(content=f"Welcome! I am a helpful assistant. How can I help you today?").send()
    #chainlit has built in message objects, and you can send them from the bot to user to see


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):

    user = cl.user_session.get("user")
    chainlit_thread_id = thread.get("id")
    logger.info(f"{user.identifier} resumed chat with thread id {chainlit_thread_id}")

    settings = cl.user_session.get("chat_settings")

    _ = await cl.ChatSettings(
        [            
            Select(
                id="LLM",
                label="OpenAI model to use",
                values=models,
                initial_value=settings['LLM']
                #whatever we have stored in LLM id previously, since we want to have it ready as the user left it
            ),

            Slider(
                id="Temperature",
                label="Temperature of the LLM",
                initial=settings["Temperature"],
                min=0,
                max=1,
                step=0.1
            )
        ]
    ).send() #puts it on screen

    cl.user_session.set("graph_builder", 
                        create_graph(
                                model=settings['LLM'],
                                temperature=settings["Temperature"],
                                tools=tools,
                                system_prompt=system_prompt))


#IMPORTANT! user settings wont be saved at all after logging out. 
# only messages they send a message/chat. our checkpoints are done only when they chat
@cl.on_message
async def on_message(msg: cl.Message):
    config = {"configurable": {"thread_id": cl.context.session.thread_id}}
    cb = cl.AsyncLangchainCallbackHandler()
    final_answer = cl.Message(content="")
    #here we just make final_answer an empty message for now to instatiate the object

    checkpointer = AsyncPostgresSaver(db_connection_pool)
    #we connect to the postgres pool here, since we want to save stuff

    # When this app runs for the first time, uncomment the following line to create database tables
    # await checkpointer.setup()
    #bc we only need to initialize the tables once at very start, then we just use them from ow on

    agent = cl.user_session.get("graph_builder").compile(checkpointer=checkpointer)
    #we want to prepare the graph for saving to postgres
  
    #keep in mind: 
    #state graphs are just blueprints, in order for them to be executable we need to make them into compiled state graphs, then the nodes can be used
    
    #our agent object is made with the checkpointer, so this streams these messages over
    async for msg, _ in agent.astream({"messages": [HumanMessage(content=msg.content)]}, 
                                    stream_mode="messages", 
                                    config=RunnableConfig(callbacks=[cb], **config)):
        #
        if (isinstance(msg, AIMessageChunk) and msg.content):
            await final_answer.stream_token(msg.content)

    await final_answer.send()


@cl.on_settings_update
async def on_settings_update(settings: dict):
    """Handle settings update"""
    user = cl.user_session.get("user")
    logger.info(f"{user.identifier} has updated settings: {settings}")
#prints that in terminial

    cl.user_session.set("graph_builder", 
                        create_graph(
                                model=settings['LLM'],
                                temperature=settings["Temperature"],
                                tools=tools,
                                system_prompt=system_prompt))
    #we set the graph with whatever is now stored in LLm id etc, we update our graph
    
    await cl.Message(content="Settings updated! How can I help you today?").send()

@cl.on_chat_end
async def on_chat_end():
    user = cl.user_session.get("user")
    chainlit_thread_id = cl.context.session.thread_id
    logger.info(f"{user.identifier} has ended the chat with thread id {chainlit_thread_id}")

@cl.on_logout
async def on_logout(request: Request, response: Response):
    ### handler to tidy up resources
    for cookie_name in request.cookies.keys():
        response.delete_cookie(cookie_name)

@cl.on_app_shutdown
async def on_app_shutdown():
    """App shutdown handler to clean up resources."""

    global db_connection_pool
    await db_connection_pool.close()
    #closing our pool
    logger.info("App shutdown: closed database connection pool")
    #prints in terminal