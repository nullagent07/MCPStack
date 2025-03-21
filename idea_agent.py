import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_deepseek import ChatDeepSeek
from langchain_core.runnables import RunnablePassthrough
from langchain.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

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

# Модель для представления декомпозированной идеи
class DecomposedIdea(BaseModel):
    original_idea: str = Field(description="Исходная идея, предложенная пользователем")
    summary: str = Field(description="Краткое описание идеи")
    tasks: List[Task] = Field(description="Список задач, на которые разбита идея")
    implementation_plan: str = Field(description="План реализации идеи")

# Инициализация модели DeepSeek
model = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.2,
    max_tokens=4000
)

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

Затем составь общий план реализации идеи, указав последовательность выполнения задач и их взаимосвязи.

{format_instructions}
"""

# Добавление инструкций по форматированию
format_instructions = parser.get_format_instructions()
decomposition_prompt = ChatPromptTemplate.from_template(decomposition_prompt_template)

# Создание цепочки для декомпозиции идеи
decomposition_chain = (
    {"idea": RunnablePassthrough(), "format_instructions": lambda _: format_instructions}
    | decomposition_prompt
    | model
    | StrOutputParser()
)

def decompose_idea(idea: str) -> Dict[str, Any]:
    """
    Декомпозирует идею на конкретные задачи.
    
    Args:
        idea: Строка с описанием идеи
        
    Returns:
        Словарь с декомпозированной идеей
    """
    try:
        # Получение структурированного ответа от модели
        result = decomposition_chain.invoke(idea)
        
        # Попытка распарсить ответ в структуру DecomposedIdea
        try:
            parsed_result = parser.parse(result)
            return parsed_result.dict()
        except Exception as e:
            # Если не удалось распарсить, возвращаем текстовый ответ
            print(f"Ошибка при парсинге ответа: {e}")
            return {"raw_response": result}
            
    except Exception as e:
        return {"error": str(e)}

def interactive_decomposition():
    """
    Интерактивный режим для декомпозиции идей.
    """
    print("=" * 50)
    print("Агент декомпозиции идей")
    print("=" * 50)
    print("Опишите вашу идею, и агент разобьет её на конкретные задачи.")
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
        
        result = decompose_idea(idea)
        
        if "error" in result:
            print(f"Произошла ошибка: {result['error']}")
            continue
            
        if "raw_response" in result:
            print(result["raw_response"])
            continue
            
        # Вывод структурированного результата
        print(f"Краткое описание: {result['summary']}\n")
        
        print("Задачи:")
        for i, task in enumerate(result['tasks'], 1):
            print(f"\n{i}. {task['name']} (Сложность: {task['estimated_complexity']})")
            print(f"   Описание: {task['description']}")
            print(f"   Инструменты: {', '.join(task['tools_needed'])}")
            
        print(f"\nПлан реализации:\n{result['implementation_plan']}")
        
        print("\n" + "-" * 50)

if __name__ == "__main__":
    interactive_decomposition()
