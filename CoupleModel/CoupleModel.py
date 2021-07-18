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
             "model":"问答对模型",
             "isError":True
             }
    showJson = {
        "view": data,
        "error": error,
    }

    return jsonify(showJson)


if __name__ == '__main__':

    app.run(port=1235, debug=True)
