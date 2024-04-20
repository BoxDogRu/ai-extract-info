# OpenAI с сервера от создателей курса
from utils import ChatOpenAI

# Схожим образом мы можем подключаться и использовать
# 1. Напрямую OpenAI, закончились токены..)
# 2. Open Source, API от HuggingFaceHub - попробовал, работает.
# 3. Open Source, локально модели с HuggingFace
# 3.1. Llama, mistral, и другие https://huggingface.co/models?pipeline_tag=text-generation&sort=trending С langchain еще не пробовал.
# 3.2. LM-Studio - это пробовал, но без langchain, отличный способ ознакомления с работой моделей
# 4. модули в langchain для других моделей https://js.langchain.com/docs/integrations/llms/
# Например YandexGPT - тут у меня есть баланс, хочу попробовать прикрутить.

import os
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm

load_dotenv()
course_api_key = os.getenv("COURSE_API_KEY")

llm = ChatOpenAI(temperature=0.0, course_api_key=course_api_key)

# пример данных для анализа
df = pd.read_csv('submission100lines.csv')

# формируем промт
prompt_template = """Ответь на вопрос используя информацию из контекста.
Если на вопрос нельзя ответить, то ответь 1.

Context: {text_input}

Question: Для скольки человек необходимо жилье?

Answer: Только число
"""

amount_list = []    # Список, где будем хранить ответы модели

for cnt, text_input in enumerate(tqdm(df['text'])):
    prompt = prompt_template.format(text_input=text_input) # Добавляем сообщение в промпт
    # try-except рекомендация в чатике курса - для случаев обрывов связи
    try:
        amount = llm.predict(prompt) # ответ модели
        amount_list.append(amount) # добавляем ответ в список
    except:
        amount_list.append(None)
    # if cnt == 0:
    #    break # Для отладки. Уберите, когда убедитесь, что на одном примере работает

df['amount'] = amount_list # Обновляем столбец из ответов модели

# преобразуем str в int (можно попробовать изменить в промте Answer: Только число int)
# во втором примере в этом нет необходимости - мы используем форматирование ответа
df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
df = df.dropna(subset=['amount'])
df['amount'] = df['amount'].astype('int64')

# сохраняем предсказания
df.to_csv('solution.csv', index=False)
