import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

BASE = os.getenv("USER_FACTS_DIR", "data/user_facts")
EMBED_MODEL = os.getenv("EMBED_MODEL")

def _get_model():
    
    _model = SentenceTransformer(EMBED_MODEL)
    return _model

def _file(chat_id: str) -> Path:
    p = Path(BASE) / f"{re.sub(r'[^\w-]', '_', str(chat_id))}.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

@tool(parse_docstring=False)
def save_fact(fact: str, category: str = "OTHER", config: RunnableConfig = None) -> str:
    """
    Сохраняет важный факт о пользователе в долговременную память.
    
    Используй, когда пользователь сообщает о себе информацию, которая может пригодиться
    в будущих диалогах для персонализации ответов.
    
    Категории (выбирай наиболее точную):
    
    - DEMO (Демография): имя, возраст, пол, город/страна проживания, часовой пояс, 
      семейное положение, наличие детей/питомцев.
      Примеры: "Пользователя зовут Анна", "Живёт в Москве", "Часовой пояс UTC+3"
    
    - PREF (Предпочтения): стиль общения, формат ответов, любимые темы, 
      что пользователь не любит в ответах, предпочтения по времени уведомлений.
      Примеры: "Предпочитает краткие ответы без воды", "Не любит эмодзи", 
      "Хочет получать напоминания вечером"
    
    - PROF (Профессия/Образование): место работы, должность, сфера деятельности, 
      учебное заведение, курс/специальность, карьерные цели.
      Примеры: "Работает backend-разработчиком", "Студент 3 курса МГУ, факультет ВМК", 
      "Планирует перейти в Data Science"
    
    - HEALTH (Здоровье/Спорт): физические параметры, график тренировок, 
      ограничения по здоровью, цели по фитнесу, режим сна/питания.
      Примеры: "Рост 179 см, вес 74 кг", "Тренируется в зале 2 раза в неделю", 
      "Не переносит лактозу", "Ложится спать до 23:00"
    
    - TECH (Технологии): языки программирования, фреймворки, инструменты, 
      ОС, предпочтения по стеку, уровень владения.
      Примеры: "Пишет на Python и C++", "Использует PyTorch для CV", 
      "Развивает навыки работы с LangChain", "Предпочитает Linux"
    
    - HOBBY (Хобби/Интересы): увлечения, спорт, музыка, книги, игры, 
      путешествия, коллекционирование, творчество.
      Примеры: "Увлекается фотографией", "Любит научную фантастику", 
      "Играет в шахматы", "Планирует поездку в Японию"
    
    - LANG (Языки): родной язык, изучаемые языки, уровень владения, 
      цели изучения, предпочтения по материалам.
      Примеры: "Родной язык — русский", "Изучает английский на уровне B1", 
      "Хочет читать технические статьи на английском"
    
    - OTHER (Прочее): важные факты, которые не подходят под другие категории.
      Используй только если категория действительно неочевидна.
    
    Правила:
    1. Сохраняй только факты, озвученные пользователем явно или с высокой уверенностью.
    2. Не сохраняй временные предпочтения ("хочу кофе сейчас") — только устойчивые.
    3. Формулируй факты в третьем лице: "Пользователь работает...", а не "Я работаю...".
    4. Один вызов — один факт. Для нескольких фактов агент сделает несколько вызовов.
    5. Если факт уже сохранён — инструмент вернёт "Duplicate", это нормально.
    
    Примеры вызова:
    - save_fact("Пользователь работает Python-разработчиком в финтехе", category="PROF")
    - save_fact("Предпочитает ответы с примерами кода на русском", category="PREF")
    - save_fact("Изучает компьютерное зрение, интересуется U-Net и YOLO", category="TECH")
    """
    chat_id = config["configurable"].get("chat_id", "default") if config else "default"
    f = _file(chat_id)
    txt = fact.strip().lower()
    if f.exists() and txt in f.read_text(encoding="utf-8").lower():
        return "Duplicate"
    with open(f, "a", encoding="utf-8") as w:
        w.write(f"[{category}] {fact} ({datetime.now():%Y-%m-%d})\n")
    return f"Saved: {fact}"

@tool(parse_docstring=False)
def get_facts(category: Optional[str] = None, config: RunnableConfig = None) -> str:
    """
    Возвращает сохранённые факты о пользователе.
    
    Используй, чтобы вспомнить контекст перед ответом на вопрос, 
    требующий знания предпочтений или истории пользователя.
    
    Параметры:
    - category: опциональная фильтрация по одной из категорий:
      DEMO, PREF, PROF, HEALTH, TECH, HOBBY, LANG, OTHER.
      Если не указана — вернутся все факты.
    
    Возвращает:
    - Отформатированный текст с фактами, сгруппированными по категориям.
    - "No facts", если ничего не найдено.
    
    Примеры:
    - get_facts() — все факты
    - get_facts(category="TECH") — только технические предпочтения
    - get_facts(category="HEALTH") — параметры и тренировки
    """
    chat_id = config["configurable"].get("chat_id", "default") if config else "default"
    f = _file(chat_id)
    if not f.exists(): return "No facts"
    facts = {}
    for line in f.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\[(\w+)\]\s+(.+?)(?:\s+\(\d{4}-\d{2}-\d{2}\))?", line.strip())
        if m: facts.setdefault(m.group(1), []).append(m.group(2))
    if category: facts = {category: facts.get(category, [])}
    out = [f"[{c}]\n" + "\n".join(f"- {i}" for i in v) for c, v in facts.items() if v]
    return "\n\n".join(out) if out else "No facts"

@tool(parse_docstring=False)
def search_facts(query: str, threshold: float = 0.6, config: RunnableConfig = None) -> str:
    """
    Ищет факты по семантической схожести с запросом.
    
    Используй, когда нужно найти факты по смыслу, а не по точному совпадению слов.
    Например: запрос "чем я занимаюсь" найдёт факты о работе и хобби.
    
    Параметры:
    - query: текст запроса на естественном языке
    - threshold: порог косинусного сходства (0.0–1.0), по умолчанию 0.6.
      Ниже 0.5 — много шума, выше 0.7 — только очень близкие совпадения.
    
    Возвращает:
    - Найденные факты с указанием категории и скором схожести.
    - "Not found", если ничего не подошло под порог.
    
    Примеры:
    - search_facts("мои параметры") → найдёт рост, вес, тренировки
    - search_facts("что я изучаю") → найдёт языки, технологии, курсы
    - search_facts("предпочтения", threshold=0.5) → широкий поиск по стилю общения
    """
    chat_id = config["configurable"].get("chat_id", "default") if config else "default"
    f = _file(chat_id)
    if not f.exists(): return "Not found"
    raw = f.read_text(encoding="utf-8").splitlines()
    lines = [l.strip() for l in raw if l.strip() and not l.startswith("#")]
    if not lines: return "Not found"
    
    texts = []
    for line in lines:
        m = re.match(r"\[(\w+)\]\s+(.+?)(?:\s+\(\d{4}-\d{2}-\d{2}\))?", line)
        texts.append(m.group(2) if m else line)
    
    model = _get_model()
    q_vec = model.encode(query, normalize_embeddings=True)
    f_vecs = model.encode(texts, normalize_embeddings=True)
    scores = np.dot(f_vecs, q_vec)
    
    matches = [(lines[i], float(scores[i])) for i in range(len(lines)) if scores[i] >= threshold]
    matches.sort(key=lambda x: x[1], reverse=True)
    
    return "\n".join(f"{l} (score: {s:.2f})" for l, s in matches) if matches else "Not found"