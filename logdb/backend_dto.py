from logdb.message import Message

class BackEndDTO:
    def __init__(self) -> None:
        self.session_id: str | None = 'ID' #Пока что захардкожено
        self.session_start_time: int | None = None
        self.session_end_time: int | None = None
        self.messages: list[Message] = []
        
    def add_message(self, message: Message) -> None:
        self.messages.append(message)