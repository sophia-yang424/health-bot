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


if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


tavily_search = TavilySearch(
    max_results=2,
    topic="general",
    include_domains=["clevelandclinic.org", "mayoclinic.org", "cdc.gov", "nih.gov", 
                    "medlineplus.gov", "medlineplus.gov", "webmd.com", "wikipedia.org"]
)

db_connection_pool: Optional[AsyncConnectionPool] = None
models = ["gpt-4o-mini"]
tools = [tavily_search]
system_prompt = ("You are a helpful health assistant. "
            "Use the Tavily Search tool to collect information to answer the user's health related questions")

logger = logging.getLogger(__name__)

def create_graph(model: str,  temperature: float, tools: list, system_prompt: str) -> CompiledStateGraph:

    base_url = os.getenv("BASE_URL")

    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        base_url=base_url,
        streaming=True)
    
    llm_with_tools = llm.bind_tools(tools)

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    graph_builder = StateGraph(State)

    async def chatbot(state: State):
        return {"messages": [await llm_with_tools.ainvoke([SystemMessage(content=system_prompt)] + state["messages"])]}

    def should_continue(state: State) -> Literal["tools", "__end__"]:
        messages = state["messages"]
        last_message = messages[-1]

        if last_message.tool_calls:
            return "tools"
        
        return "__end__"

    tool_node = ToolNode(tools)

    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_conditional_edges(
        "chatbot",
        should_continue,
        {"tools": "tools", "__end__": END}
    )
    graph_builder.add_edge("tools", "chatbot")

    return graph_builder


@cl.on_app_startup
async def on_app_startup():
    """App startup handler to initialize resources."""

    global db_connection_pool
    db_connection_pool = AsyncConnectionPool(os.getenv("LANGGRAPH_DB_URL"), open=False,
                                            min_size=1, max_size=8,
                                            kwargs={
                                                        "autocommit": True,
                                                        "row_factory": dict_row,
                                                    }
                                            )
    await db_connection_pool.open(wait=True)
    logger.info("App startup: initialized database connection pool")

@cl.data_layer
def get_data_layer():
    return SQLAlchemyDataLayer(conninfo=os.getenv("DATABASE_URL"))

@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
    """Password auth handler for login"""
    
    # For development only, DO NOT use in production.
    # Secure authentication and authorization methods are necessary for production deployment
    if username == "admin": 
        return cl.User(identifier="admin", metadata={"role": "ADMIN"})
    else: 
        return None

@cl.on_chat_start
async def on_chat_start():
    user = cl.user_session.get("user")
    chainlit_thread_id = cl.context.session.thread_id
    logger.info(f"{user.identifier} started a new chat with thread id {chainlit_thread_id}")

    settings = await cl.ChatSettings(
        [            
            Select(
                id="LLM",
                label="OpenAI model to use",
                values=models,
                initial_value=models[0]
            ),

            Slider(
                id="Temperature",
                label="Temperature of the LLM",
                initial=0,
                min=0,
                max=1,
                step=0.1
            )
        ]
    ).send()

    cl.user_session.set("graph_builder", 
                        create_graph(
                                model=settings['LLM'],
                                temperature=settings["Temperature"],
                                tools=tools,
                                system_prompt=system_prompt))
    
    await cl.Message(content=f"Welcome! I am a helpful assistant. How can I help you today?").send()


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
    ).send()

    cl.user_session.set("graph_builder", 
                        create_graph(
                                model=settings['LLM'],
                                temperature=settings["Temperature"],
                                tools=tools,
                                system_prompt=system_prompt))


@cl.on_message
async def on_message(msg: cl.Message):
    config = {"configurable": {"thread_id": cl.context.session.thread_id}}
    cb = cl.AsyncLangchainCallbackHandler()
    final_answer = cl.Message(content="")

    checkpointer = AsyncPostgresSaver(db_connection_pool)

    # When this app runs for the first time, uncomment the following line to create database tables
    # await checkpointer.setup()

    agent = cl.user_session.get("graph_builder").compile(checkpointer=checkpointer)
    
    async for msg, _ in agent.astream({"messages": [HumanMessage(content=msg.content)]}, 
                                    stream_mode="messages", 
                                    config=RunnableConfig(callbacks=[cb], **config)):
        
        if (isinstance(msg, AIMessageChunk) and msg.content):
            await final_answer.stream_token(msg.content)

    await final_answer.send()


@cl.on_settings_update
async def on_settings_update(settings: dict):
    """Handle settings update"""
    user = cl.user_session.get("user")
    logger.info(f"{user.identifier} has updated settings: {settings}")

    cl.user_session.set("graph_builder", 
                        create_graph(
                                model=settings['LLM'],
                                temperature=settings["Temperature"],
                                tools=tools,
                                system_prompt=system_prompt))
    
    await cl.Message(content="Settings updated! How can I help you today?").send()

@cl.on_chat_end
async def on_chat_end():
    user = cl.user_session.get("user")
    chainlit_thread_id = cl.context.session.thread_id
    logger.info(f"{user.identifier} has ended the chat with thread id {chainlit_thread_id}")

@cl.on_logout
async def on_logout(request: Request, response: Response):
    ### Handler to tidy up resources
    for cookie_name in request.cookies.keys():
        response.delete_cookie(cookie_name)

@cl.on_app_shutdown
async def on_app_shutdown():
    """App shutdown handler to clean up resources."""

    global db_connection_pool
    await db_connection_pool.close()
    logger.info("App shutdown: closed database connection pool")