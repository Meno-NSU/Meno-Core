from datetime import datetime
from datetime import timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from meno_core.infrastructure.logdb.backend_dto import BackEndDTO
from meno_core.infrastructure.logdb.model.db_models import Conversation, Turn


class LogCollector:
    def __init__(self):
        # TODO: Возможно, стоит переименовать
        self._unreleased_dtos: dict[str, BackEndDTO] = {}

        self._db_url = 'postgresql://logdb:123@localhost:5432/logdb'  # пока захардкодил
        self._db_engine = create_engine(self._db_url, echo=True)

        Session = sessionmaker(bind=self._db_engine)  # тут надо с менеджером контекстным
        self._session = Session()

        self._test_counter = 0

    def _add_to_db(self, session_id: str, dto: BackEndDTO) -> None:
        conversation = Conversation(user_id=session_id, start_time=datetime.utcnow(), end_time=datetime.utcnow())

        messages = dto.get_messages()
        for i in range(len(messages)):
            msg = messages[i]
            turn = Turn(turn=i, original_question=msg.get_question(), expanded_question=msg.get_expanded_question(),
                        resolved_question=msg.get_coref_resolved_question(), answer=msg.get_answer())
            conversation.turns.append(turn)

        self._session.add(conversation)
        self._session.commit()

    def create_message(self, session_id: str):
        delta = timedelta(seconds=30)

        if session_id in self._unreleased_dtos.keys():
            delta = timedelta(seconds=30)
            dto = self._unreleased_dtos[session_id]

            start_time = dto._session_start_time  # не круто
            end_time = dto._session_end_time

            if abs(start_time - end_time) >= delta:
                self._add_to_db(session_id=session_id, dto=dto)
                self._unreleased_dtos.pop(session_id)

        if session_id not in self._unreleased_dtos.keys():
            self._unreleased_dtos[session_id] = BackEndDTO(session_id=session_id, start_time=datetime.utcnow())

        self._unreleased_dtos[session_id].create_empty_message()

    def add_question(self, session_id: str, text: str):
        self._unreleased_dtos[session_id].add_question(text)

    def add_expanded_question(self, session_id: str, text: str):
        self._unreleased_dtos[session_id].add_expanded_question(text)

    def add_resolved_question(self, session_id: str, text: str):
        self._unreleased_dtos[session_id].add_resolved_question(text)

    def add_model_answer(self, session_id: str, text: str):
        self._unreleased_dtos[session_id].add_answer(text)

    def update_time(self, session_id: str):
        self._unreleased_dtos[session_id]._session_end_time = datetime.utcnow()

    # потестить
    def print_dto(self, session_id: str):
        dto = self._unreleased_dtos[session_id]
        for i in range(len(dto._messages)):
            print(dto._session_start_time)
            print(dto._session_end_time)
            print('DEFAULT:\n', dto._messages[i]._question)
            print('EXPANDED:\n', dto._messages[i]._expanded_question)
            print('RESOLVED:\n', dto._messages[i]._coref_resolved_question)
            print('ANSWER:\n', dto._messages[i]._model_answer)
