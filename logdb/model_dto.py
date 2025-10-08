from dataclasses import *



@dataclass
class ModelDTO:
    user_id: int
    session_start_time: int 
    session_end_time: int

    
