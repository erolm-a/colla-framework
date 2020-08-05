from flask import Flask, send_file, redirect, session
from flask_restful import Resource, Api, reqparse, fields, marshal_with
from flask_restful_swagger import swagger
import pickle

from tools.answering import QuestionAnsweringContext, SerializedIntent, failed_intent

app = Flask(__name__)
api = swagger.docs(Api(app),
                   apiVersion='0.1',
                   description="A Basic Chatbot API")

app.secret_key = b'\x97\xba\xbb\xe4\xac5\x94\x10c\xda\x82\xb0\x9d\xe9\xa2|'

@swagger.model
class IntentResponse:
    resource_fields = {
        'intentType': fields.String,
        'message': fields.String
    }

@swagger.model
class Utterance:
    resource_fields = {
        'utterance': fields.String
    }

class DialogueAnsweringResource(Resource):
    @swagger.operation(
        notes="""Create a session. The call leaves a FLASK_SESSION
        token that expires after 24 hours, or explicitely deleted.""",
        responseClass=IntentResponse.__name__
    )
    @marshal_with(IntentResponse.resource_fields)
    def get(self):
        session['context'] = pickle.dumps(QuestionAnsweringContext())
        return SerializedIntent(SerializedIntent.IntentType.WELCOME, 'Hi, this is the Grill-Lab chatbot! This context will terminate in 24 hours.')
    
    @swagger.operation(
        notes="Reply to an user utterance.",
        responseClass=IntentResponse.__name__,
        parameters=[
            {
                "name": "utterance",
                "description": "The user utterance",
                "required": True,
                "allowMultiple": False,
                "dataType": Utterance.__name__,
                "paramType": "body"
            }
        ],

    )
    @marshal_with(IntentResponse.resource_fields)
    def post(self):
        if not 'context' in session:
            return failed_intent('Context was not created')

        context = pickle.loads(session['context'])

        argparser = reqparse.RequestParser()
        argparser.add_argument('utterance')
        args = argparser.parse_args()

        response = context.handle_question(args['utterance'])
        session['context'] = pickle.dumps(context)

        return response

        
api.add_resource(DialogueAnsweringResource, "/api/chat")

@app.route("/")
def serve_frontend():
    return send_file('build/index.html')

@app.route("/docs")
def docs():
    return redirect("/static/docs.html")

if __name__ == '__main__':
    app.run(debug=True)

