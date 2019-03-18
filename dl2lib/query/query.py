from .parser import Parser
from . import api as q

class Query:

    def __init__(self, query, context, args):
        self.query = query
        self.context = context
        self.args = args
        self.constraint, self.return_values = Parser(query, self.context, self.args).parse()

    def run(self):
        return q.solve(self.constraint, self.args, return_values=self.return_values)
