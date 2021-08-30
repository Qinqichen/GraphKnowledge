import flask
from flask import request , Blueprint
import json
from .ModelClass import ModelClass 


deepModelController_Blueprint = Blueprint("deepModelController_Blueprint", __name__ )


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


# 将预测函数定义为一个端点
@deepModelController_Blueprint.route("/getDeepModelData", methods=["GET","POST"])
def DeepModel():
    json_data = request.get_json()
    print(json_data)

    '''
    {#深度模型请求参数
        "question":"青花缠枝牡丹纹罐是什么朝代得？"
                }
    '''
    
    
    return doDeepModel(json_data)

def doDeepModel(json_data):

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
        return json.dumps(showJson)
    
    for entity in entitys:
        data[entity['type']] = entity['entity']
    
    # 关系检测
    relations,errorRelation = modelClass.doModelFunction("测试问题",2)
    if errorRelation['isError'] == True :
        showJson['error'] = errorRelation
        return json.dumps(showJson)
    
    for relation in relations:
        data['relation'] = relation['relation']
    
    
    # TODO: 测试代码，后期要修改回
    
    error = {"id":-1,
             "type":"",
             "description":"深度模型测试错误",
             "model":"深度模型",
             "isError":True
             }
    
    showJson = {
        "view": data,
        "error": error,
    }

    return json.dumps(showJson)
