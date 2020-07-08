from flask import Flask, send_file, redirect
from flask_restful import Resource, Api
from flask_restful_swagger import swagger

from tools.answering import parse_natural_question

app = Flask(__name__)
api = swagger.docs(Api(app), apiVersion='0.1')

class HelloWorld(Resource):
    def get(self):
        return {"hello": "world"}


class QuestionAnsweringResource(Resource):
    @swagger.operation(
        notes="""Answer a natural question. This call is not meant to be used in a dialogue-like situation where previous context is needed."""
    )
    def get(self, natural_question):
        """Answer context-less natural language questions."""

        senses = parse_natural_question(natural_question)
        return {"intentType": "expressionDefinition",
                "senses": senses
                }
        

api.add_resource(HelloWorld, "/api")
api.add_resource(QuestionAnsweringResource, "/api/query/<string:natural_question>")

@app.route("/")
def serve_frontend():
    return send_file('build/index.html')

@app.route("/docs")
def docs():
    return redirect("/static/docs.html")

if __name__ == '__main__':
    app.run(debug=True)

