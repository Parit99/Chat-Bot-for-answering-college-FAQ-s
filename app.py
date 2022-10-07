from check_bot import *
from flask import Flask, render_template, request
from new_chatbot import *
official='If the reponse got is insatisfactory you can check your query at official website https://www.pdpu.ac.in/'

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    param='cosine'
    userText = request.args.get('msg')
    ans,flag=get_bot_resp(userText,param)
    if(ans!=None):
        return ans+official
    else:
        return official
if __name__ == '__main__':
	app.run()