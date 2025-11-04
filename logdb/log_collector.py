from logdb.backend_dto import BackEndDTO

class LogCollector:
    def __init__(self):
        #Возможно, стоит переименовать
        self._unreleased_dtos: dict[str, BackEndDTO] = {}

    def add_question(self, session_id: str, text: str):
        #TODO: тут нужна лочка(скорее всего)
        if not(session_id in self._unreleased_dtos.keys()):
            self._unreleased_dtos[session_id] = BackEndDTO(session_id=session_id)
        
        self._unreleased_dtos[session_id].add_question(text)

        


