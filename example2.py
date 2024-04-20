import os
from dotenv import load_dotenv

from utils import ChatOpenAI

from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

load_dotenv()
course_api_key = os.getenv("COURSE_API_KEY")

llm = ChatOpenAI(temperature=0.0, course_api_key=course_api_key)

customer_review = """
Этот фен для волос просто потрясающий. Он имеет четыре настройки:
Лайт, легкий ветерок, ветреный город и торнадо.
Он прибыл через два дня, как раз к приезду моей жены -
подарок на годовщину.
Думаю, моей жене это настолько понравилось, что она потеряла дар речи.
Этот фен немного дороже, чем другие но я думаю,
что дополнительные функции того стоят.
"""

# Понадобится ещё одна сущность: схема ответа - ResponseSchema
gift_schema = ResponseSchema(name="gift",
                             description="Был ли товар куплен в подарок кому-то другому? Ответь «True», если да, «False», если нет или неизвестно.")

delivery_days_schema = ResponseSchema(name="delivery_days",
                                      description="Сколько дней потребовалось для доставки товара? Если эта информация не найдена, выведи -1.")

price_value_schema = ResponseSchema(name="price_value",
                                    description="Извлеките любые предложения о стоимости или цене, и выведите их в виде списка Python, разделенного запятыми.")

response_schemas = [gift_schema,
                    delivery_days_schema,
                    price_value_schema]

# Создаём парсер и подаём в него список со схемами
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# получаем инструкции по форматированию ответа
format_instructions = output_parser.get_format_instructions()

# шаблон, внизу добавляем инструкции для форматирования
review_template = """\
Из следующего текста извлеки информацию:

gift: Был ли товар куплен в подарок кому-то другому?
Ответь «True», если да, «False», если нет или неизвестно.

delivery_days: Сколько дней потребовалось для доставки товара? 
Если эта информация не найдена, выведи -1.

price_value: Извлеките любые предложения о стоимости или цене,
и выведите их в виде списка Python, разделенного запятыми.

text: {text}

{format_instructions}

"""

prompt = ChatPromptTemplate.from_template(template=review_template)

messages = prompt.format_messages(text=customer_review,
                                format_instructions=format_instructions)

response = llm.invoke(messages)
output_dict = output_parser.parse(response.content) # преобразуем ответ в словарь

print(output_dict)  # {'gift': 'True', 'delivery_days': '2', 'price_value': 'Этот фен немного дороже, чем другие'}

print(output_dict.get('gift'))  # True
