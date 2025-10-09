class Message:
    def __init__(self) -> None:
        self.question: str | None = None
        self.expanded_question: str | None = None
        self.coref_resolved_question: str | None = None
        self.model_answer : str | None = None

    def set_question(self, question: str) -> None:
        self.question = question

    def set_expanded_question(self, question: str) -> None:
        self.expanded_question = question

    def set_coref_resolved_question(self, question: str) -> None:
        self.coref_resolved_question = question    

    def set_model_answer(self, answer: str) -> None:
        self.model_answer = answer
