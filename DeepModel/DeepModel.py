import flask
import pandas as pd
import torch
from flask import request,jsonify
from ModelClass import ModelClass 

# 测试
"""
data = {
        "id": 1,
        "name" : "深度模型视图",
        "origin" : "测试问题",
        "object" : "青花缠枝牡丹纹罐 ",
        "subject": "朝代",
        "relation": "是什么时间",
   }
    
modelClass = ModelClass()

enitys = modelClass.doModelFunction("测试问题",1)
for enity in enitys:
    data[enity['type']] = enity['enity']

relations = modelClass.doModelFunction("测试问题",2)
for relation in relations:
    data['relation'] = relation['relation']
    
print(data)
"""
# 测试结束




# 实例化 flask
app = flask.Flask(__name__)

# 加载模型
#model = torch.load('modelname')

# 将预测函数定义为一个端点
@app.route("/getDeepModelData", methods=["GET","POST"])
def DeepModel():
    json_data = request.get_json()
    print(json_data)

    '''
    {#深度模型请求参数
        "question":"青花缠枝牡丹纹罐是什么朝代得？"
                }
    '''
    
    data = {
        "id": 1,
        "name" : "深度模型视图",
        "origin" : json_data['question'],
        "object" : "",
        "subject": "",
        "relation": "",
    }
    
    showJson = {
            "view": data,
            "error": ''
        }
    
    
    # 模型使用类
    modelClass = ModelClass()
    
    # 实体识别
    entitys,errorEntity = modelClass.doModelFunction("测试问题",1)
    if errorEntity['isError'] == True :
        showJson['error'] = errorEntity
        return jsonify(showJson)
    
    for entity in entitys:
        data[entity['type']] = entity['entity']
    
    # 关系检测
    relations,errorRelation = modelClass.doModelFunction("测试问题",2)
    if errorRelation['isError'] == True :
        showJson['error'] = errorRelation
        return jsonify(showJson)
    
    for relation in relations:
        data['relation'] = relation['relation']
    
    
    error = {"id":-1,
             "type":"",
             "description":"",
             "model":"深度模型",
             "isError":False
             }
    
    showJson = {
        "view": data,
        "error": error,
    }

    return jsonify(showJson)

