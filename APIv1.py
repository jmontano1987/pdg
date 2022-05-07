################Librerias utilizadas########################
from flask import Flask, request, make_response            #
import json                                                #
import functions as func                                   #
import pickle                                              #
############################################################

app = Flask(__name__)

model_rf,tfidf = func.load_models()

@app.route('/')
def hello():
    return 'Hello World.'

#Metodo POST
@app.route('/webhook', methods=['POST'])

def webhook():
    req = request.get_json(silent=True, force=True)
    res = processRequest(req)
    res = json.dumps(res)
    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'
    return r  #Final Response sent to DialogFlow

#Procesa las solicitudes recibidas por Dialogflow
def processRequest(req):
    result = req.get("queryResult")
    parameters = result.get("parameters")
    news=parameters.get("noticia")
    
    intent = result.get("intent").get('displayName')
    
#Valida si el usuario selecciono RandomForest para la clasificación
    if (intent=='RandomForest'):
        output=func.detection_fake_news_(news,model_rf,tfidf)   
        fulfillmentText= " {} !".format(output)
        return {
            "fulfillmentText": fulfillmentText
        }
#Valida si el usuario selecciono el mejor modelo para la clasificación
    if (intent=='Menu Principal - Best Model'):
        output=func.detection_fake_news_(news,model_rf,tfidf)   
        fulfillmentText= " {} !".format(output)
        return {
            "fulfillmentText": fulfillmentText
        }

if __name__ == '__main__':
    app.run()