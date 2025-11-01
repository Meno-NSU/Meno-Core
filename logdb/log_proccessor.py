from logdb.backend_dto import BackEndDTO

class LogProcessor:
    def __init__(self):
        #Возможно, стоит переименовать
        self.unreleased_dtos: dict = {}

    def update_db(self, dto: BackEndDTO):
        if (dto in self.unreleased_dtos):
            #тут отправлять в бд
            return

        self.unreleased_dtos[dto.session_id] = dto

        


