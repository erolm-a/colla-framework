from flask import Flask, send_file, redirect, session
from flask_restful import Resource, Api, reqparse
from flask_restful_swagger import swagger

from tools.answering import answer_natural_question, QuestionAnsweringContext

app = Flask(__name__)
api = swagger.docs(Api(app), apiVersion='0.1')

app.secret_key = b'\x97\xba\xbb\xe4\xac5\x94\x10c\xda\x82\xb0\x9d\xe9\xa2|'

class HelloWorld(Resource):
    def get(self):
        return {"hello": "world"}


class QuestionAnsweringResource(Resource):
    @swagger.operation(
        notes="""Answer a natural question. This call is not meant to be used
        in a dialogue-like situation where previous context is needed."""
    )
    def get(self, natural_question):
        """Answer context-less natural language questions."""

        senses = answer_natural_question(natural_question)
        return {"intentType": "expressionDefinition",
                "senses": senses
                }
        
class DialogueAnsweringResource(Resource):
    @swagger.operation(
        notes="""Create a session. The call leaves a FLASK_SESSION
        token that expires after 24 hours, or explicitely deleted."""
    )
    def get(self):
        session['context'] = QuestionAnsweringContext()
        return {'welcomeText': 'Hi, this is the Grill-Lab chatbot! This context will terminate in 24 hours.'}
    
    @swagger.operation(
        notes="Reply to an user utterance."
    )
    def post(self):
        if not 'context' in session:
            return {'response': 'Context was not created.'}, 401

        context = session['context']

        argparser = reqparse.RequestParser()
        argparser.add_argument('utterance')
        args = argparser.parse_args()

        return context.handle_question(args['utterance'])
        
api.add_resource(HelloWorld, "/api")
api.add_resource(QuestionAnsweringResource, "/api/query/<string:natural_question>")
api.add_resource(DialogueAnsweringResource, "/api/chat")

@app.route("/")
def serve_frontend():
    return send_file('build/index.html')

@app.route("/docs")
def docs():
    return redirect("/static/docs.html")

if __name__ == '__main__':
    app.run(debug=True)

