from dataclasses import *


@dataclass
class UserDTO:
    user_id: int
    messages: list[tuple]
    session_start_time: int
    session__end_time: int
    