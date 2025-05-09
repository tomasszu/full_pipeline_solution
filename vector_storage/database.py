from opensearch_logic import Opensearch_db


class Database:
    def __init__(self):

        self.db = Opensearch_db("localhost", 9200, ("admin", "admin"))

    def insert(self, id, vector):

        self.db.insert(id, vector)

    def query(self, vector):

        return self.db.query_vector(vector)