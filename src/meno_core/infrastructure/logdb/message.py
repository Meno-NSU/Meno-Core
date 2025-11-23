
class Message:
    def __init__(self) -> None:
        self._question: str | None = None
        self._expanded_question: str | None = None
        self._coref_resolved_question: str | None = None
        self._model_answer : str | None = None

    def set_question(self, question: str) -> None:
        self._question = question

    def set_expanded_question(self, question: str) -> None:
        self._expanded_question = question

    def set_coref_resolved_question(self, question: str) -> None:
        self._coref_resolved_question = question    

    def set_model_answer(self, answer: str) -> None:
        self._model_answer = answer

    def get_question(self) -> str | None:
        return self._question    
    
    def get_expanded_question(self) -> str | None:
        return self._expanded_question
    
    def get_coref_resolved_question(self) -> str | None:
        return self._coref_resolved_question 
    
    def get_answer(self) -> str | None:
        return self._model_answer