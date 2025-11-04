from logdb.backend_dto import BackEndDTO

class LogCollector:
    def __init__(self):
        #TODO: Возможно, стоит переименовать
        self._unreleased_dtos: dict[str, BackEndDTO] = {}

    def add_question(self, session_id: str, text: str):
        #TODO: тут нужна лочка(скорее всего)
        if not(session_id in self._unreleased_dtos.keys()):
            self._unreleased_dtos[session_id] = BackEndDTO(session_id=session_id)
        
        self._unreleased_dtos[session_id].add_question(text)

    def add_expanded_question(self, session_id: str, text: str):
        self._unreleased_dtos[session_id].add_expanded_question(text)

    def add_resolved_question(self, session_id: str, text: str):
        self._unreleased_dtos[session_id].add_resolved_question(text)

    #потестить
    def print_dto(self, session_id: str):
        dto = self._unreleased_dtos[session_id]
        for i in range(dto._messages_size + 1): 
            print('DEFAULT:\n',dto._messages[i]._question)
            print('EXPANDED:\n', dto._messages[i]._expanded_question)
            print('RESOLVED:\n', dto._messages[i]._coref_resolved_question)



