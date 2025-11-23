import uuid
from datetime import datetime

from sqlalchemy import DateTime, Text, UUID, Integer, ForeignKey
from sqlalchemy import String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Turn(Base):
    __tablename__ = "turns"

    conversation_id: Mapped[UUID] = mapped_column(UUID, primary_key=True, default=uuid.uuid4)

    conversation_id: Mapped[UUID] = mapped_column(
        UUID, 
        ForeignKey('conversations.conversation_id'),
        primary_key=True
    )

    turn: Mapped[int] = mapped_column(Integer, primary_key=True)
    
    original_question: Mapped[str] = mapped_column(Text, nullable=True) 
    expanded_question: Mapped[str] = mapped_column(Text, nullable=True) 
    resolved_question: Mapped[str] = mapped_column(Text, nullable=True)
    answer: Mapped[str] = mapped_column(Text, nullable=True)

    conversation: Mapped["Conversation"] = relationship("Conversation", back_populates="turns")


class Conversation(Base):
    __tablename__ = "conversations"

    conversation_id: Mapped[UUID] = mapped_column(UUID, primary_key=True, default=uuid.uuid4)

    user_id: Mapped[str] = mapped_column(String, nullable=True)

    start_time: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    end_time: Mapped[datetime] = mapped_column(DateTime, nullable=True)
 
    turns: Mapped[list["Turn"]] = relationship("Turn", back_populates="conversation")


    