# agent.py
import os
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig  # ← добавили
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from tools.tg import send_telegram_tool
from tools.search import search_tool
from tools.calendar import (
    add_event_tool,
    list_events_tool,
    parse_relative_date_tool,
    get_current_datetime_tool,
)
from tools.reminders import create_reminder_tool
from utils.reminders import start_reminder_checker
from tools.weather import get_weather_tool
from utils.memory import LangChainMemory
from pathlib import Path
load_dotenv()
PROMPTS_DIR = Path(__file__).parent / "prompts"

TOOLS = [
    search_tool,
    add_event_tool,
    parse_relative_date_tool,
    get_current_datetime_tool,
    list_events_tool,
    send_telegram_tool,
    create_reminder_tool,
    get_weather_tool
]

llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "google/gemma-4-26b-a4b-it:free"),
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.0,
)
llm_with_tools = llm.bind_tools(TOOLS)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def load_prompt(filename: str) -> str:
    """Просто читает .txt файл из папки prompts."""
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8").strip()

def build_system_prompt(long_term: str | None = None) -> str:
    """Собирает системный промпт. Если есть память — подставляет её в шаблон."""
    base = load_prompt("system_base.txt")
    if not long_term:
        return base
    
    memory = load_prompt("system_memory.txt")
    return f"{base}\n\n{memory.format(long_term=long_term)}"
def call_model(state: AgentState, config: RunnableConfig) -> dict:
    """Чистый вызов LLM. Память и системный промпт уже в state["messages"]."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}



def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


# Сборка графа
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(TOOLS))
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")

checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)


# agent.py (замени функцию run_agent на эту)

def run_agent(user_query: str, session_id: str = "default", chat_id: int = -1) -> str:
    config = {"configurable": {"thread_id": session_id, "chat_id": chat_id}}
    mem = LangChainMemory(session_id)
    
    # 1️⃣ Ищем релевантную долгосрочную историю (1 API-вызов на эмбеддинг)
    long_term = mem.search_long_term(user_query)
    
    # 2️⃣ Собираем финальный системный промпт
    system_content = build_system_prompt(long_term)
    # Добавляем тех. инфу для тулзов (напоминания, отправка в TG)
    system_content += f"\n\nID текущего чата пользователя = `{chat_id}`. Используй его при создании напоминаний или отправке сообщений."
    
    initial_messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=user_query)
    ]
    
    # 3️⃣ Запускаем граф
    result = app.invoke({"messages": initial_messages}, config=config)
    
    # 4️⃣ Сохраняем ТОЛЬКО новые сообщения в долгосрочную память (дельта)
    new_msgs = result["messages"][len(initial_messages):]
    if new_msgs:
        mem.add_messages(new_msgs)
        
    return result["messages"][-1].content


