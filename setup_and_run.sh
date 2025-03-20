#!/bin/bash
python -m venv venv

# Активация виртуального окружения
source venv/bin/activate

# Установка необходимых библиотек
pip install langchain-mcp-adapters langgraph langchain-openai

# Запуск вашего скрипта
python math_server.py

# Деактивация виртуального окружения
deactivate
