import os
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Optional
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from langchain_core.tools import tool  # ← добавить в импорты

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SCOPES = ["https://www.googleapis.com/auth/calendar"]

BASE_DIR = Path(__file__).parent
CREDENTIALS_PATH = BASE_DIR / "credentials.json"
TOKEN_PATH = BASE_DIR / "token.json"         # Создаётся автоматически после первой авторизации
TIMEZONE = "Europe/Moscow"                   # Укажи свой часовой пояс

class AddEventInput(BaseModel):
    summary: str = Field(description="Название события")
    start: str = Field(description="Начало в формате YYYY-MM-DD HH:MM")
    end: str = Field(description="Конец в формате YYYY-MM-DD HH:MM")
    description: Optional[str] = Field(default=None, description="Описание (опционально)")

class ListEventsInput(BaseModel):
    limit: int = Field(default=5, ge=1, le=20, description="Макс. кол-во событий для вывода")
class ParseRelativeDateInput(BaseModel):
    relative: str = Field(description="Относительная дата: 'завтра', 'через 3 дня', 'в понедельник', 'на следующей неделе'")
    base_date: Optional[str] = Field(default=None, description="Базовая дата в формате YYYY-MM-DD (по умолчанию — сегодня)")
    time: Optional[str] = Field(default=None, description="Время в формате HH:MM (опционально, добавляется к результату)")

def _get_calendar_service():
    """Инициализирует Google Calendar API с кэшированием токена."""
    creds = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CREDENTIALS_PATH.exists():
                raise FileNotFoundError(
                    "❌ Не найден credentials.json.\n"
                    "👉 Скачай его: https://developers.google.com/calendar/api/quickstart/python#set_up_your_environment"
                )
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)  # Откроет браузер для авторизации
        with open(TOKEN_PATH, "w") as f:
            f.write(creds.to_json())
            
    return build("calendar", "v3", credentials=creds)
@tool
def get_current_datetime_tool() -> str:
    """Возвращает текущую дату и время. Используй, когда пользователь говорит 'сегодня', 'завтра', 'через час' и т.п. Формат: YYYY-MM-DD HH:MM (часовой пояс: {TIMEZONE})."""
    try:
        now = datetime.now()
        return json.dumps({
            "current_datetime": now.strftime("%Y-%m-%d %H:%M"),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M"),
            "weekday": now.strftime("%A"),
            "timezone": TIMEZONE
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Ошибка получения времени: {str(e)}"}, ensure_ascii=False)
@tool
def parse_relative_date_tool(relative: str, base_date: Optional[str] = None, time: Optional[str] = None) -> str:
    """Парсит относительные даты на русском. 
    Поддерживает: 'сегодня', 'завтра', 'вчера', 'через N дней/недель/месяцев', 'назад', 'в понедельник/вторник/...', 
    'на этой/следующей/прошлой неделе', 'в начале/конце месяца'. 
    Возвращает дату в формате YYYY-MM-DD или YYYY-MM-DD HH:MM (если указано время)."""
    try:
        # Базовая дата
        if base_date:
            base = datetime.strptime(base_date, "%Y-%m-%d")
        else:
            base = datetime.now()
        
        relative = relative.strip().lower()
        result_date = base

        # 📅 Простые случаи
        if relative in ("сегодня", "сейчас", "текущий день"):
            pass
        elif relative in ("завтра", "следующий день"):
            result_date = base + timedelta(days=1)
        elif relative in ("послезавтра"):
            result_date = base + timedelta(days=2)
        elif relative in ("вчера", "предыдущий день"):
            result_date = base - timedelta(days=1)
        elif relative in ("позавчера"):
            result_date = base - timedelta(days=2)

        # 🔢 "через N дней/недель/месяцев/лет"
        elif match := re.match(r"через\s+(\d+)\s+(дн[еяй]|день|дня|недел[юи]|недель|мес[яи]ц[ае]?|месяц[ае]?|лет|год[аи]?|года?)", relative):
            n = int(match.group(1))
            unit = match.group(2)
            if "недел" in unit:
                result_date = base + timedelta(weeks=n)
            elif "мес" in unit:
                # Прибавляем месяцы с учётом переполнения
                month = base.month + n
                year = base.year + (month - 1) // 12
                month = (month - 1) % 12 + 1
                day = min(base.day, [31,29 if year%4==0 and (year%100!=0 or year%400==0) else 28,31,30,31,30,31,31,30,31,30,31][month-1])
                result_date = base.replace(year=year, month=month, day=day)
            elif "лет" in unit or "год" in unit:
                try:
                    result_date = base.replace(year=base.year + n)
                except ValueError:  # 29 Feb → non-leap year
                    result_date = base.replace(year=base.year + n, day=28)
            else:  # дни по умолчанию
                result_date = base + timedelta(days=n)

        # 🔙 "N дней назад", "неделю назад"
        elif match := re.match(r"(\d+)?\s*(дн[еяй]?|день|дня|недел[юи]|недель|мес[яи]ц[ае]?|месяц[ае]?|лет|год[аи]?|года?)\s*назад", relative):
            n = int(match.group(1)) if match.group(1) else 1
            unit = match.group(2)
            if "недел" in unit:
                result_date = base - timedelta(weeks=n)
            elif "мес" in unit:
                month = base.month - n
                year = base.year - (1 - month) // 12
                month = (month - 1) % 12 + 1
                day = min(base.day, [31,29 if year%4==0 and (year%100!=0 or year%400==0) else 28,31,30,31,30,31,31,30,31,30,31][month-1])
                result_date = base.replace(year=year, month=month, day=day)
            elif "лет" in unit or "год" in unit:
                try:
                    result_date = base.replace(year=base.year - n)
                except ValueError:
                    result_date = base.replace(year=base.year - n, day=28)
            else:
                result_date = base - timedelta(days=n)

        # 🗓 "в понедельник", "в следующий вторник" и т.д.
        elif match := re.match(r"(в\s+)?(следующ[ийую]|прошл[ыйую]|этой|на этой|на следующей|на прошлой)?\s*(понедельник|вторник|сред[ау]|четверг|пятниц[ау]|суббот[ау]|воскресень[ея])", relative):
            weekday_map = {"понедельник": 0, "вторник": 1, "среда": 2, "среду": 2, "четверг": 3, "пятница": 4, "пятницу": 4, "суббота": 5, "субботу": 5, "воскресенье": 6, "воскресенье": 6}
            target_wd = weekday_map.get(match.group(3), 0)
            modifier = match.group(2)
            current_wd = base.weekday()
            
            if modifier in ("прошл", "на прошлой"):
                days_diff = (target_wd - current_wd - 7) % 7 - 7
            elif modifier in ("следующ", "на следующей"):
                days_diff = (target_wd - current_wd + 7) % 7
                if days_diff == 0:
                    days_diff = 7
            else:  # "в понедельник", "этой неделе"
                days_diff = (target_wd - current_wd + 7) % 7
            result_date = base + timedelta(days=days_diff)

        # 📆 "на этой/следующей/прошлой неделе"
        elif "неделе" in relative:
            if "прошл" in relative:
                result_date = base - timedelta(weeks=1)
            elif "следующ" in relative:
                result_date = base + timedelta(weeks=1)
            # "на этой неделе" → оставляем base

        # 🌙 "в начале/конце месяца"
        elif "начале месяца" in relative:
            result_date = base.replace(day=1)
        elif "конце месяца" in relative:
            # Последний день месяца
            if base.month == 12:
                result_date = base.replace(year=base.year+1, month=1, day=1) - timedelta(days=1)
            else:
                result_date = base.replace(month=base.month+1, day=1) - timedelta(days=1)

        # ❓ Нераспознано → пробуем парсить как абсолютную дату
        else:
            try:
                result_date = datetime.strptime(relative, "%Y-%m-%d")
            except ValueError:
                return json.dumps({"error": f"Не удалось распарсить дату: '{relative}'. Поддерживаемые форматы: 'завтра', 'через 3 дня', 'в понедельник', 'на следующей неделе'"}, ensure_ascii=False)

        # Добавляем время, если указано
        if time:
            try:
                t = datetime.strptime(time, "%H:%M")
                result_date = result_date.replace(hour=t.hour, minute=t.minute)
                return json.dumps({"date": result_date.strftime("%Y-%m-%d %H:%M"), "date_only": result_date.strftime("%Y-%m-%d"), "time": result_date.strftime("%H:%M")}, ensure_ascii=False)
            except ValueError:
                pass  # Игнорируем неверный формат времени

        return json.dumps({"date": result_date.strftime("%Y-%m-%d"), "date_only": result_date.strftime("%Y-%m-%d")}, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": f"Ошибка парсинга даты: {str(e)}"}, ensure_ascii=False)
@tool
def add_event_tool(summary: str, start: str, end: str, description: Optional[str] = None) -> str:
    """Добавляет событие в Google Calendar (primary календарь пользователя)."""
    try:
        service = _get_calendar_service()
        dt_start = datetime.strptime(start, "%Y-%m-%d %H:%M")
        dt_end = datetime.strptime(end, "%Y-%m-%d %H:%M")

        event = {
            "summary": summary,
            "description": description or "",
            "start": {"dateTime": dt_start.isoformat(), "timeZone": TIMEZONE},
            "end": {"dateTime": dt_end.isoformat(), "timeZone": TIMEZONE},
        }
        created = service.events().insert(calendarId="primary", body=event).execute()
        return json.dumps({
            "status": "ok",
            "id": created["id"],
            "link": created.get("htmlLink", ""),
            "message": f"Событие '{summary}' создано в Google Calendar"
        }, ensure_ascii=False)
    except ValueError as e:
        return json.dumps({"error": f"Неверный формат даты. Ожидается YYYY-MM-DD HH:MM. {e}"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Google Calendar ошибка: {str(e)}"}, ensure_ascii=False)
@tool
def delete_event_tool(
    summary: str, 
    start: str, 
    end: str, 
    description: Optional[str] = None,
    exact_match: bool = False  # Новый параметр: строгое совпадение названия
) -> str:
    """Удаляет событие из Google Calendar по названию и временному окну.
    
    Args:
        summary: Название события (поиск по подстроке, если exact_match=False)
        start: Начало временного окна для поиска в формате YYYY-MM-DD HH:MM
        end: Конец временного окна для поиска в формате YYYY-MM-DD HH:MM
        description: Дополнительный фильтр по описанию (опционально)
        exact_match: Если True — удалять только при точном совпадении названия
    
    Returns:
        JSON со статусом операции и данными удалённого события
    """
    try:
        service = _get_calendar_service()
        
        # Парсим временное окно для поиска
        dt_start = datetime.strptime(start, "%Y-%m-%d %H:%M")
        dt_end = datetime.strptime(end, "%Y-%m-%d %H:%M")
        
        # 1️⃣ Ищем события в указанном окне
        events_result = service.events().list(
            calendarId="primary",
            timeMin=dt_start.isoformat(),
            timeMax=dt_end.isoformat(),
            singleEvents=True,
            orderBy="startTime",
            maxResults=50  # Ограничиваем выборку для безопасности
        ).execute()
        
        items = events_result.get("items", [])
        if not items:
            return json.dumps({
                "error": f"В окне {start} — {end} не найдено ни одного события"
            }, ensure_ascii=False)
        
        # 2️⃣ Фильтруем по названию (и опционально по описанию)
        candidates = []
        for event in items:
            event_summary = event.get("summary", "")
            event_desc = event.get("description", "")
            
            # Проверка названия
            name_match = (event_summary == summary) if exact_match else (summary.lower() in event_summary.lower())
            # Проверка описания (если указано)
            desc_match = (description is None) or (description.lower() in (event_desc or "").lower())
            
            if name_match and desc_match:
                candidates.append(event)
        
        if not candidates:
            return json.dumps({
                "error": f"Не найдено событий с названием '{summary}' в указанном окне. "
                        f"Доступные события: {[e.get('summary') for e in items]}"
            }, ensure_ascii=False)
        
        # 3️⃣ Если найдено несколько — удаляем первое (или можно вернуть ошибку)
        if len(candidates) > 1:
            return json.dumps({
                "warning": f"Найдено {len(candidates)} подходящих событий. Удалено первое.",
                "deleted": {
                    "summary": candidates[0].get("summary"),
                    "start": candidates[0]["start"].get("dateTime", candidates[0]["start"].get("date")),
                    "id": candidates[0]["id"]
                },
                "others": [
                    {"summary": e.get("summary"), "start": e["start"].get("dateTime", e["start"].get("date"))}
                    for e in candidates[1:]
                ]
            }, ensure_ascii=False)
        
        # 4️⃣ Удаляем событие по ID
        target = candidates[0]
        service.events().delete(calendarId="primary", eventId=target["id"]).execute()
        
        return json.dumps({
            "status": "ok",
            "deleted": {
                "summary": target.get("summary"),
                "start": target["start"].get("dateTime", target["start"].get("date")),
                "id": target["id"]
            },
            "message": f"Событие '{target.get('summary')}' успешно удалено"
        }, ensure_ascii=False)
        
    except ValueError as e:
        return json.dumps({
            "error": f"Неверный формат даты. Ожидается YYYY-MM-DD HH:MM. {e}"
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "error": f"Google Calendar ошибка: {str(e)}"
        }, ensure_ascii=False)
@tool
def list_events_tool(limit: int = 5) -> str:
    """Возвращает ближайшие события из Google Calendar."""
    try:
        service = _get_calendar_service()
        now = datetime.utcnow().isoformat() + "Z"
        
        result = service.events().list(
            calendarId="primary",
            timeMin=now,
            maxResults=limit,
            singleEvents=True,
            orderBy="startTime"
        ).execute()
        
        items = result.get("items", [])
        out = []
        for e in items:
            start = e["start"].get("dateTime", e["start"].get("date"))
            out.append({
                "title": e.get("summary", ""),
                "start": start,
                "link": e.get("htmlLink", "")
            })
        return json.dumps(out, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Google Calendar ошибка: {str(e)}"}, ensure_ascii=False)