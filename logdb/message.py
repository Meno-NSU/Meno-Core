class Message:
    def __init__(self) -> None:
        self.question = None
        self.expanded_question = None
        self.coref_resolved_question = None
        self.model_answer = None

    def set_question(self, question: str) -> None:
        self.question = question

    def set_expanded_question(self, question: str) -> None:
        self.expanded_question = question

    def coref_resolved_question(self, question: str) -> None:
        self.coref_resolved_question = question    
        