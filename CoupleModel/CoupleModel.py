import flask
import pandas as pd
import torch
from flask import request,jsonify

# 实例化 flask
app = flask.Flask(__name__)

# 加载模型
#model = torch.load('modelname')

# 将预测函数定义为一个端点
@app.route("/getCoupleModelData", methods=["GET","POST"])
def doCoupleModel():
    json_data = request.get_json()
    print(json_data)

    '''
    
    {#问答对模型请求参数
        "question":"青花缠枝牡丹纹罐是什么朝代得？"
                }

    '''

    data = {
         "id": 2,
        "name": "问答对模型视图",
        "question": "青花缠枝牡丹纹罐是什么朝代得？",
        "answer": "问答对模型的数据库答案",
        "have":False
    }
    
    
    
    error = { "id":2,
             "type":"nullError",
             "description":"无数据",
             "model":"问答对模型测试",
             "isError":True
             }
    # 测试语句
    if json_data['question'] == '问答对模型有答案' :
        
        data['hava'] = True
        
        error['isError'] = False
    
    showJson = {
        "view": data,
        "error": error,
    }

    return jsonify(showJson)

# 将预测函数定义为一个端点
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
