import flask
from flask import request,jsonify 
from .DatabaseController import DatabaseController as DBController
import time
import logging

handler = logging.FileHandler("flask.log")

# 实例化 flask
app = flask.Flask(__name__)


app.logger.addHandler(handler)

# 将预测函数定义为一个端点
@app.route("/getCoupleModelData", methods=["GET","POST"])
def doCoupleModel():
    json_data = request.get_json()
    
    
    # json_data = {
    #         'question': 'ttt'
    #     }

    '''
    
    {#问答对模型请求参数
        "question":"青花缠枝牡丹纹罐是什么朝代得？"
                }

    '''
    data = {
        "id": 2,
        "name": "问答对模型视图",
        "question": json_data['question'],
        "answer": "",
        "have":False
    }
    error = { "id":-1,
             "type":"",
             "description":"",
             "model":"问答对模块",
             "isError":False
             }
    startTime = time.time()
    # TODO:待修改 添加错误类型
    dataFromDB = DBController.QuestionCoupleDB().selectByQuestion(json_data)
    app.logger.warning("coupleModel内部调用 Time: "+str(time.time() - startTime))
    data['answer'] = dataFromDB['answer']
    data['have'] = dataFromDB['have']
    
    
    
    showJson = {
        "view": data,
        "error": error,
    }

    return jsonify(showJson)

@app.route("/saveQuestionCouple", methods=["GET","POST"])
def saveQuestionCouple():
    json_data = request.get_json()
    print(json_data)

    '''
    json_data = {
        "question":"",
        "answer":""
                }

    '''

    # 调用数据库存储功能函数
    # 例如： r = DataBasePool.QuestionCouple.insert(json_data['question'],json_data['answer'])
    # 
   
  

    return jsonify("保存成功")


if __name__ == '__main__':
    app.run(port=1234, debug=True)