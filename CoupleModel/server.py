from flask import Flask,render_template,request
from flask_sqlalchemy import SQLAlchemy
import config

app = Flask(__name__)  # 创建Flask对象
app.config.from_object(config)  # 关联config.py文件进来
db = SQLAlchemy(app)  # 建立和数据库的关系映射

class QA(db.Model):
    __tablename__ = 'qa'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    question = db.Column(db.String)
    answer = db.Column(db.String)

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



if __name__ == '__main__':
    app.run(port=1234, debug=True)