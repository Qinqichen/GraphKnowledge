import flask
import pandas as pd
import requests
import torch
from flask import request,jsonify

# 实例化 flask
app = flask.Flask(__name__)

# 加载模型
#model = torch.load('modelname')

# 将预测函数定义为一个端点
@app.route("/getKnowledgeModelData", methods=["GET","POST"])
def KnowledgeModel():
    json_data = request.get_json()
    print(json_data)

    '''
    {#图谱查询模型请求参数
        "id": 1,
        "name" : "深度模型视图",
        "origin" : "青花缠枝牡丹纹罐是什么朝代得？",
        "object" : "青花缠枝牡丹纹罐 ",
        "subject": "朝代",
        "relation": "是什么时间",
    }
    '''
    data = { #需返回数据
        "id": 3,
        "name": "知识图谱模型视图",
        "CQL": "MATCH (n:object)-[r1:object_time]->(m:time)",
        "return": "n: 青花缠枝牡丹纹罐, m:元朝",
        "answer":"知识图谱模型答案", #新加字段，存放最终答案
    }
    error = {"id":-1,
             "type":"",
             "description":"",
             "model":"图谱查询模型",
             "isError":False
             }
    
    # 测试语句
    if json_data['origin'] == '图谱模型出错' :
        
        error['isError'] = True
    
    
    
    showJson = {
        "view": data,
        "error": error,
    }

    return jsonify(showJson)
if __name__ == '__main__':

    app.run(port=1235, debug=True)
