from logdb.message import Message
from datetime import datetime

class BackEndDTO:
    def __init__(self, session_id: str, start_time: datetime) -> None:
        self._session_id: str | None = session_id
        self._session_start_time: int | None = start_time
        self._session_end_time: datetime = None
        self._messages: list[Message] = []
            
    def create_empty_message(self) -> None:
        self._messages.append(Message())

    def add_question(self, text: str) -> None:
        #TODO: здесь нужен лок
        self._messages[len(self._messages) - 1].set_question(text)

    def add_expanded_question(self, text: str) -> None:
        self._messages[len(self._messages) - 1].set_expanded_question(text)
        
    def add_resolved_question(self, text: str) -> None:
        self._messages[len(self._messages) - 1].set_coref_resolved_question(text)

    def add_answer(self, text: str) -> None:
        self._messages[len(self._messages) - 1].set_model_answer(text)