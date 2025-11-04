
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