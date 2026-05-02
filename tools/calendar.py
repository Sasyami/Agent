import os
import json
from datetime import datetime
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