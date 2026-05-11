import os
from pathlib import Path
from typing import Annotated, Literal, Sequence

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.runnables import RunnableConfig

from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama

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
from tools.get_info import (
    save_fact,
    get_facts,
    search_facts,
    
)
from tools.reminders import (
    cancel_reminder_tool,
    create_reminder_tool,
)
from tools.weather import get_weather_tool

from utils.memory import LangChainMemory




load_dotenv()

PROMPTS_DIR = Path(__file__).parent / "prompts"

MAX_REFLECTION_ITERATIONS = 3




TOOLS = [
    search_tool,
    add_event_tool,
    parse_relative_date_tool,
    get_current_datetime_tool,
    list_events_tool,
    send_telegram_tool,
    create_reminder_tool,
    cancel_reminder_tool,
    get_weather_tool,
    save_fact,
    get_facts,
    search_facts,
]




llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME"),
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.2,
)

llm_with_tools = llm.bind_tools(TOOLS)




reflect_llm = ChatOllama(
    model=os.getenv("REFLECT_MODEL", "qwen2.5:7b"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    temperature=0.1,
)
reviser_llm = ChatOllama(
    model=os.getenv("REVISER_MODEL", "qwen2.5:3b"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    temperature=0.1,
)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

    critique: str
    reflection_decision: str

    iteration: int

    long_term_memory: str




class ReflectionResult(BaseModel):
    decision: Literal["accept", "revise"] = Field(
        description="Нужно ли исправлять ответ"
    )

    critique: str = Field(
        description="Конкретная критика ответа"
    )


reflect_chain = reflect_llm.with_structured_output(
    ReflectionResult
)



def load_prompt(filename: str) -> str:
    return (PROMPTS_DIR / filename).read_text(
        encoding="utf-8"
    ).strip()


def build_system_prompt(
    long_term: str | None = None
) -> str:

    base = load_prompt("system_base.txt")

    if not long_term:
        return base

    memory = load_prompt("system_memory.txt")

    return f"""
{base}

{memory.format(long_term=long_term)}
""".strip()


def call_model(
    state: AgentState,
    config: RunnableConfig,
):

    response = llm_with_tools.invoke(
        state["messages"],
        config=config,
    )

    return {
        "messages": [response]
    }


def route_after_agent(state: AgentState):

    last = state["messages"][-1]

    if getattr(last, "tool_calls", None):
        return "tools"

    return "reflect"



def reflect(
    state: AgentState,
    config: RunnableConfig,
):

    last_answer = state["messages"][-1].content

    chat_id = (
        config.get("configurable", {})
        .get("chat_id", "default")
    )

    facts = get_facts(chat_id)

    prompt = f"""
Ты — строгий критик расписаний и планов.

Нужно проверить:

- перегрузку
- отсутствие отдыха
- плохой баланс задач
- нереалистичность
- неудобство
- конфликты
- логические проблемы

Оцени качество ответа.

Если есть серьезные проблемы:
decision = revise

Если ответ хороший:
decision = accept

Факты пользователя:
{facts if facts else "нет"}

Ответ:
{last_answer}
"""

    result = reflect_chain.invoke(prompt)

    return {
        "critique": result.critique,
        "reflection_decision": result.decision,
        "iteration": state.get("iteration", 0) + 1,
    }




def route_after_reflect(state: AgentState):

    iteration = state.get("iteration", 0)

    if iteration >= MAX_REFLECTION_ITERATIONS:
        return END

    decision = state["reflection_decision"]

    if decision == "accept":
        return END

    return "revise"



def revise(
    state: AgentState,
    config: RunnableConfig,
):

    critique = state["critique"]

    last_answer = state["messages"][-1].content

    prompt = f"""
Ты создаешь краткую инструкцию
для улучшения ответа другого ИИ.

Ответ:
{last_answer}

Критика:
{critique}

Сформулируй:
- что исправить
- что сохранить
- что улучшить

Коротко и конкретно.

Не переписывай ответ.
Не объясняй.
Верни только instruction.
"""

    revision_instruction = reviser_llm.invoke([
        SystemMessage(content=prompt)
    ]).content

    return {
        "messages": [
            SystemMessage(
                content=f"""
Исправь предыдущий ответ.

Инструкция:
{revision_instruction}
"""
            )
        ]
    }



workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)

workflow.add_node(
    "tools",
    ToolNode(TOOLS),
)

workflow.add_node("reflect", reflect)

workflow.add_node("revise", revise)


workflow.set_entry_point("agent")




workflow.add_conditional_edges(
    "agent",
    route_after_agent,
)




workflow.add_edge(
    "tools",
    "agent",
)



workflow.add_conditional_edges(
    "reflect",
    route_after_reflect,
)



workflow.add_edge(
    "revise",
    "agent",
)



checkpointer = MemorySaver()

app = workflow.compile(
    checkpointer=checkpointer
)



def run_agent(
    user_query: str,
    session_id: str = "default",
    chat_id: int = -1,
) -> str:

    config = {
        "configurable": {
            "thread_id": session_id,
            "chat_id": str(chat_id),
        }
    }

    mem = LangChainMemory(session_id)

    long_term = mem.search_long_term(user_query)

    system_content = build_system_prompt(
        long_term
    )

    system_content += f"""

ID текущего чата пользователя = `{chat_id}`.

Используй его при:
- создании напоминаний
- отправке сообщений
"""

    initial_messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=user_query),
    ]

    result = app.invoke(
        {
            "messages": initial_messages,
            "iteration": 0,
        },
        config=config,
    )

    new_msgs = result["messages"][
        len(initial_messages):
    ]

    if new_msgs:
        mem.add_messages(new_msgs)

    return result["messages"][-1].content