from logdb.message import Message

class BackEndDTO:
    def __init__(self, session_id: str) -> None:
        self._session_id: str | None = session_id
        self._session_start_time: int | None = None
        self._session_end_time: int | None = None
        self._start_count = 20
        self._messages: list[Message] = [Message() for _ in range(self._start_count)]
        self._messages_size: int = 0
            
    def add_question(self, text: str) -> None:
        #проверка работоспособности:
        #TODO: здесь нужен лок
        self._messages[self._messages_size].set_question(text)
        
        
    def add_expanded_question(self, text: str) ->None:
        self._messages[self._messages_size].set_expanded_question(text)
        #временно:
        self._messages_size += 1
    
        