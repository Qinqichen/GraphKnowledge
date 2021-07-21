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
    
    modelClass = ModelClass()
    
    enitys = modelClass.doModelFunction("测试问题",1)
    for enity in enitys:
        data[enity['type']] = enity['enity']
    
    relations = modelClass.doModelFunction("测试问题",2)
    for relation in relations:
        data['relation'] = relation['relation']
    
    
    error = {"id":1,
             "type":"timeOut",
             "description":"模型超时",
             "model":"深度模型",
             "isError":False
             }
    
    # 测试语句
    if json_data['question'] == '深度模型出错' :
        
        error['isError'] = True
    # 测试完毕
    
    showJson = {
        "view": data,
        "error": error,
    }

    return jsonify(showJson)




# if __name__ == '__main__':

#     app.run(port=1235, debug=True)
