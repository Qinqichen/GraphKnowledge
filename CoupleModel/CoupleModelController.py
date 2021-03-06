
from flask import request,jsonify ,Blueprint
from .DatabaseController import DatabaseController as DBController
import json

coupleModel_Blueprint = Blueprint("coupleModel", __name__)


# 将函数定义为一个端点
@coupleModel_Blueprint.route("/getCoupleModelData", methods=["GET","POST"])
def doCoupleModel(json_data):
    # json_data = request.get_json()
    
    
    # json_data = {
    #         'question': 'aaa'
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
    
    # TODO:待修改 添加错误类型
    dataFromDB = DBController.QuestionCoupleDB().selectByQuestion(json_data)
    data['answer'] = dataFromDB['answer']
    data['have'] = dataFromDB['have']
    
    
    
    showJson = {
        "view": data,
        "error": error,
    }

    return json.dumps(showJson)

@coupleModel_Blueprint.route("/saveQuestionCouple", methods=["GET","POST"])
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
