from flask import Flask,render_template,request,jsonify,Blueprint
from flask_sqlalchemy import SQLAlchemy
import json

import sys
sys.path.append('..')
from init import db


questionDB_Blueprint = Blueprint("questionDB", __name__ )

# questionDB_Blueprint.config.from_object(config)  # 关联config.py文件进来

# db = SQLAlchemy(questionDB_Blueprint)  # 建立和数据库的关系映射

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



@questionDB_Blueprint.route('/')
def index():
    return render_template('login.html')

@questionDB_Blueprint.route('/login', methods = ["POST"])
def login():
    username = request.form.get("username")
    password = request.form.get("pwd")
    if username == 'aaa' and password == '123':
        return render_template('index.html')
    else:
        return render_template('login.html', msg="error")
#查询操作
@questionDB_Blueprint.route('/select/<string:a>')  # 跳转测试。
def select(a):
    qa = QA.query.filter(QA.question == a).first()
    question = qa.question
    answer = qa.answer
    #print(qa)
    print(question,answer)
    #return answer
    return render_template('test.html',question = question, answer = answer)

#增
@questionDB_Blueprint.route('/add')
def add():
    qa = QA(question='1adad', answer='1adada')
    db.session.add(qa)
    db.session.qa()
    return "success"

#删
@questionDB_Blueprint.route('/delete')
def delete():
    a = QA.query.filter(QA.question == "这是问题1").first()
    db.session.delete(a)
    db.session.qa()
    return "success"

#改
@questionDB_Blueprint.route('/a')
def a():
    qa = QA.query.filter(QA.question == "djakda").first()
    qa.question = '888'
    db.session.qa()
    return "success"


# 用问题查询答案代码
@questionDB_Blueprint.route('/SelectByQuestion/<string:question>')
def SelectByQuestion(question):
    
    result = {
        'question':question,
        'answer':'',
        'have':False
        }
    
    questionAnswer = QuestionCouple.query.filter(QuestionCouple.question == question).first()
    
    if questionAnswer != None:
        result['answer'] = questionAnswer.answer
        result['have'] = True
        
    else:
        # 可以加入其他功能
        
        pass
        
    return json.dumps(result)
        

