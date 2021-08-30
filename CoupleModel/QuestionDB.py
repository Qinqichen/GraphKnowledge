from flask import Flask,render_template,request,jsonify
from flask_sqlalchemy import SQLAlchemy
import json
from . import config
import time
import logging

handler = logging.FileHandler("flask.log")


app = Flask(__name__)  # 创建Flask对象
app.config.from_object(config)  # 关联config.py文件进来
db = SQLAlchemy(app)  # 建立和数据库的关系映射

app.logger.addHandler(handler)
class QA(db.Model):
    __tablename__ = 'qa'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    question = db.Column(db.String)
    answer = db.Column(db.String)

class QuestionCouple(db.Model):
    
    __tablename__ = 'questionCouple'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    question = db.Column(db.String(512),unique = True ,index = True)
    answer = db.Column(db.String(512))
    queryNum = db.Column(db.Integer,default = 0 )
    useful = db.Column(db.Integer,default = 0 )
    unuseful = db.Column(db.Integer,default = 0 )



@app.route('/')
def index():
    return render_template('login.html')

@app.route('/login', methods = ["POST"])
def login():
    username = request.form.get("username")
    password = request.form.get("pwd")
    if username == 'aaa' and password == '123':
        return render_template('index.html')
    else:
        return render_template('login.html', msg="error")
#查询操作
@app.route('/select/<string:a>')  # 跳转测试。
def select(a):
    qa = QA.query.filter(QA.question == a).first()
    question = qa.question
    answer = qa.answer
    #print(qa)
    print(question,answer)
    #return answer
    return render_template('test.html',question = question, answer = answer)

#增
@app.route('/add')
def add():
    qa = QA(question='1adad', answer='1adada')
    db.session.add(qa)
    db.session.qa()
    return "success"

#删
@app.route('/delete')
def delete():
    a = QA.query.filter(QA.question == "这是问题1").first()
    db.session.delete(a)
    db.session.qa()
    return "success"

#改
@app.route('/a')
def a():
    qa = QA.query.filter(QA.question == "djakda").first()
    qa.question = '888'
    db.session.qa()
    return "success"


# 用问题查询答案代码
@app.route('/SelectByQuestion')
def SelectByQuestion():
    startTime = time.time()
    question = request.get_json()['question']
    
    result = {
        'question':question,
        'answer':'',
        'have':False
        }
    
    questionAnswer = QuestionCouple.query.filter(QuestionCouple.question == question).first()
    
    if questionAnswer != None:
        result['answer'] = questionAnswer.answer
        result['have'] = True
        app.logger.warning("查询数据库 Time: "+str(time.time() - startTime))
        return jsonify(result)
    else:
        # 可以加入其他功能
        
        app.logger.warning("查询数据库 Time: "+str(time.time() - startTime))
        return json.dumps(result)
        

# end  qqc add 



if __name__ == '__main__':
    app.run(port=1234, debug=True)