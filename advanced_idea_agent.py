import os
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_deepseek import ChatDeepSeek
from langchain_core.runnables import RunnablePassthrough
from langchain.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import Tool
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START
from typing_extensions import TypedDict

# Загрузка переменных окружения
load_dotenv()

# Проверка наличия API ключа DeepSeek
if not os.getenv("DEEPSEEK_API_KEY"):
    raise ValueError("DEEPSEEK_API_KEY не найден в переменных окружения. Пожалуйста, добавьте его в .env файл.")

# Модель для представления задачи
class Task(BaseModel):
    name: str = Field(description="Название задачи")
    description: str = Field(description="Подробное описание задачи")
    tools_needed: List[str] = Field(description="Список инструментов или API, необходимых для выполнения задачи")
    estimated_complexity: str = Field(description="Оценка сложности задачи (низкая, средняя, высокая)")
    category: str = Field(description="Категория задачи (например, 'сбор данных', 'анализ', 'разработка', 'публикация')")

# Модель для представления декомпозированной идеи
class DecomposedIdea(BaseModel):
    original_idea: str = Field(description="Исходная идея, предложенная пользователем")
    summary: str = Field(description="Краткое описание идеи")
    tasks: List[Task] = Field(description="Список задач, на которые разбита идея")
    implementation_plan: str = Field(description="План реализации идеи")

# Определение состояния для графа
class AgentState(TypedDict):
    idea: str
    decomposed_idea: Optional[DecomposedIdea]
    research_results: Dict[str, Any]
    current_task: Optional[str]
    final_report: Optional[str]

# Инициализация модели DeepSeek
model = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.2,
    max_tokens=4000
)

# Инструменты для агента
@tool
def search_web(query: str) -> str:
    """Поиск информации в интернете по заданному запросу."""
    search_tool = TavilySearchResults(max_results=3)
    try:
        results = search_tool.invoke(query)
        return str(results)
    except Exception as e:
        return f"Ошибка при поиске: {str(e)}"

@tool
def search_wikipedia(query: str) -> str:
    """Поиск информации в Википедии по заданному запросу."""
    wiki = WikipediaAPIWrapper()
    try:
        return wiki.run(query)
    except Exception as e:
        return f"Ошибка при поиске в Википедии: {str(e)}"

@tool
def generate_code_snippet(task_description: str) -> str:
    """Генерирует код на основе описания задачи."""
    code_prompt = ChatPromptTemplate.from_template(
        "Напиши код для решения следующей задачи:\n\n{task_description}\n\n"
        "Предоставь полный код с комментариями и объяснением."
    )
    code_chain = code_prompt | model | StrOutputParser()
    return code_chain.invoke({"task_description": task_description})

# Создание парсера для структурированного вывода
parser = PydanticOutputParser(pydantic_object=DecomposedIdea)

# Промпт для декомпозиции идеи
decomposition_prompt_template = """
Ты - опытный аналитик и разработчик. Твоя задача - помочь пользователю декомпозировать его идею на конкретные задачи, 
которые нужно выполнить для её реализации.

Идея пользователя:
{idea}

Проанализируй эту идею и разбей её на логические задачи. Для каждой задачи определи:
1. Название задачи
2. Подробное описание того, что нужно сделать
3. Необходимые инструменты или API
4. Оценку сложности (низкая, средняя, высокая)
5. Категорию задачи (например, 'сбор данных', 'анализ', 'разработка', 'публикация')

Задачи должны быть определены исходя из контекста идеи. Они не фиксированы и могут включать:
- Сбор и парсинг данных
- Обращение к базам данных
- Поиск информации (например, в Википедии)
- Создание человекочитаемого текста
- Написание кода
- Публикация результатов
- Любые другие задачи, необходимые для реализации идеи

Затем составь общий план реализации идеи, указав последовательность выполнения задач и их взаимосвязи.

{format_instructions}
"""

# Добавление инструкций по форматированию
format_instructions = parser.get_format_instructions()
decomposition_prompt = ChatPromptTemplate.from_template(decomposition_prompt_template)

# Функции для графа состояний
def decompose_idea_step(state: AgentState) -> AgentState:
    """Шаг декомпозиции идеи"""
    idea = state["idea"]
    
    # Формирование промпта
    prompt_input = {"idea": idea, "format_instructions": format_instructions}
    prompt_result = decomposition_prompt.invoke(prompt_input)
    
    # Получение ответа от модели
    model_result = model.invoke(prompt_result)
    result_text = model_result.content
    
    # Парсинг результата
    try:
        decomposed = parser.parse(result_text)
        return {"decomposed_idea": decomposed}
    except Exception as e:
        print(f"Ошибка при парсинге: {e}")
        # Возвращаем сырой текст, если парсинг не удался
        return {"decomposed_idea": {"original_idea": idea, "summary": "Ошибка парсинга", 
                                   "tasks": [], "implementation_plan": result_text}}

def research_tasks(state: AgentState) -> AgentState:
    """Шаг исследования задач"""
    decomposed_idea = state.get("decomposed_idea")
    if not decomposed_idea:
        return {"research_results": {}}
    
    research_results = {}
    
    for task in decomposed_idea.tasks:
        if "сбор данных" in task.category.lower() or "поиск" in task.category.lower():
            # Выполняем поиск для задач, связанных со сбором данных
            search_query = f"{decomposed_idea.summary} {task.name}"
            web_results = search_web(search_query)
            wiki_results = search_wikipedia(task.name)
            
            research_results[task.name] = {
                "web_search": web_results,
                "wikipedia": wiki_results
            }
    
    return {"research_results": research_results}

def generate_code_for_tasks(state: AgentState) -> AgentState:
    """Шаг генерации кода для задач"""
    decomposed_idea = state.get("decomposed_idea")
    if not decomposed_idea:
        return {"code_snippets": {}}
    
    code_snippets = {}
    
    for task in decomposed_idea.tasks:
        if "разработка" in task.category.lower() or "код" in task.category.lower():
            code = generate_code_snippet(f"{decomposed_idea.summary} - {task.description}")
            code_snippets[task.name] = code
    
    return {"code_snippets": code_snippets}

def create_final_report(state: AgentState) -> AgentState:
    """Создание финального отчета"""
    decomposed_idea = state.get("decomposed_idea")
    research_results = state.get("research_results", {})
    code_snippets = state.get("code_snippets", {})
    
    if not decomposed_idea:
        return {"final_report": "Не удалось декомпозировать идею."}
    
    report = f"""
# Отчет по идее: {decomposed_idea.summary}

## Исходная идея
{decomposed_idea.original_idea}

## Краткое описание
{decomposed_idea.summary}

## Задачи
"""
    
    for i, task in enumerate(decomposed_idea.tasks, 1):
        report += f"""
### {i}. {task.name} (Сложность: {task.estimated_complexity}, Категория: {task.category})
**Описание:** {task.description}
**Необходимые инструменты:** {', '.join(task.tools_needed)}
"""
        
        # Добавление результатов исследования, если есть
        if task.name in research_results:
            report += f"""
**Результаты исследования:**
- Веб-поиск: {research_results[task.name].get('web_search', 'Нет данных')}
- Википедия: {research_results[task.name].get('wikipedia', 'Нет данных')}
"""
        
        # Добавление сгенерированного кода, если есть
        if task.name in code_snippets:
            report += f"""
**Сгенерированный код:**
```python
{code_snippets[task.name]}
```
"""
    
    report += f"""
## План реализации
{decomposed_idea.implementation_plan}
"""
    
    return {"final_report": report}

# Создание графа состояний
workflow = StateGraph(AgentState)

# Добавление узлов
workflow.add_node("decompose_idea", decompose_idea_step)
workflow.add_node("research_tasks", research_tasks)
workflow.add_node("generate_code", generate_code_for_tasks)
workflow.add_node("create_report", create_final_report)

# Определение потока
workflow.add_edge(START, "decompose_idea")
workflow.add_edge("decompose_idea", "research_tasks")
workflow.add_edge("research_tasks", "generate_code")
workflow.add_edge("generate_code", "create_report")

# Компиляция графа
agent_executor = workflow.compile()

def process_idea(idea: str) -> Dict[str, Any]:
    """
    Обрабатывает идею с использованием агента.
    
    Args:
        idea: Строка с описанием идеи
        
    Returns:
        Словарь с результатами обработки
    """
    try:
        # Запуск графа состояний
        result = agent_executor.invoke({"idea": idea})
        return result
    except Exception as e:
        return {"error": str(e)}

def interactive_mode():
    """
    Интерактивный режим для обработки идей.
    """
    print("=" * 50)
    print("Продвинутый агент декомпозиции идей")
    print("=" * 50)
    print("Опишите вашу идею, и агент разобьет её на конкретные задачи,")
    print("проведет исследование и предложит код для реализации.")
    print("Для выхода введите 'exit' или 'quit'.")
    print("-" * 50)
    
    while True:
        idea = input("\nВаша идея: ")
        
        if idea.lower() in ['exit', 'quit']:
            print("До свидания!")
            break
            
        if not idea.strip():
            print("Пожалуйста, введите идею.")
            continue
            
        print("\nАнализирую вашу идею...\n")
        
        result = process_idea(idea)
        
        if "error" in result:
            print(f"Произошла ошибка: {result['error']}")
            continue
        
        # Вывод финального отчета
        if "final_report" in result:
            print(result["final_report"])
        else:
            print("Не удалось создать отчет.")
        
        print("\n" + "-" * 50)

if __name__ == "__main__":
    interactive_mode()
