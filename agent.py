import os
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# 🔹 Импорт тулзов из твоих файлов (они теперь с @tool)
from tools.search import search_tool
from tools.schedule_parser import parse_schedule_tool
from tools.calendar import add_event_tool, list_events_tool

load_dotenv()

# 🔹 1. Список тулзов (просто перечисляем функции)
TOOLS = [search_tool, parse_schedule_tool, add_event_tool, list_events_tool]

# 🔹 2. Инициализация LLM с биндингом тулзов
llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "google/gemma-4-26b-a4b-it:free"),
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.0
)
llm_with_tools = llm.bind_tools(TOOLS)

# 🔹 3. Определение состояния графа
class AgentState(TypedDict):
    """Состояние агента: просто список сообщений с автоматическим слиянием."""
    messages: Annotated[Sequence[BaseMessage], add_messages]

# 🔹 4. Узел: вызов LLM
def call_model(state: AgentState):
    # Добавляем system-промпт в начало
    system_msg = SystemMessage(content="Ты ассистент. Используй инструменты строго по необходимости. Отвечай кратко и по делу на русском.")
    messages = [system_msg] + [m for m in state["messages"] if not isinstance(m, SystemMessage)]
    
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}  # LangGraph сам сольёт с историей

# 🔹 5. Роутер: решать, идти в тулзы или завершать
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    # Если у сообщения есть tool_calls → идём в ToolNode, иначе → конец
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

# 🔹 6. Сборка графа
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)                    # Узел: LLM
workflow.add_node("tools", ToolNode(TOOLS))               # Узел: выполнение тулзов (автоматический!)
workflow.set_entry_point("agent")                         # Старт с LLM
workflow.add_conditional_edges("agent", should_continue, {
    "tools": "tools",   # Если есть tool_calls → идём в ToolNode
    END: END            # Иначе → завершаем
})
workflow.add_edge("tools", "agent")                       # После тулзов → снова в LLM

# Компилируем граф в исполняемое приложение
app = workflow.compile()

# 🔹 7. Удобная функция запуска
def run_agent(user_query: str, session_id: str = "default", max_steps: int = 10) -> str:
    """Запускает агента с новым запросом."""
    # Для простоты: каждый запуск — с чистого листа
    # (если нужна история — подключи checkpointer или HybridMemory из прошлого ответа)
    
    result = app.invoke(
        {"messages": [HumanMessage(content=user_query)]},
        config={"recursion_limit": max_steps}  # Защита от бесконечных циклов
    )
    # Последний элемент в messages — финальный ответ
    return result["messages"][-1].content

# 🔹 Точка входа
if __name__ == "__main__":
    query = "Добавь питание в календарь сегодня 02.05.2026 на 19:00. Потом покажи список."
    print(f"👤 Запрос: {query}")
    print(f"💡 Ответ агента:\n{run_agent(query)}")