from flask import Flask
from flask_restful import Resource, Api
from flask_restful_swagger import swagger

app = Flask(__name__)
api = swagger.docs(Api(app), apiVersion='0.1')

class HelloWorld(Resource):
    def get(self):
        return {"hello": "world"}

api.add_resource(HelloWorld, "/api")

if __name__ == '__main__':
    app.run(debug=True)

