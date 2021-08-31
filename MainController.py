# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 11:09:43 2021

@author: 秦琦琛

@pythonVersion: python-3.8
"""

from flask import Flask , request ,jsonify,Blueprint , current_app
import requests
import ConfigKnowledge as cfgG
import json
import time

import CoupleModel.CoupleModelController as coupleMC
import DeepModel.DeepModel as deepMC
import KnowledgeModel.KnowledgeModelController as knowledgeMC

mainController_Blueprint = Blueprint("mainController", __name__ )


# 调用问答对模型获取答案
# /getCoupleModelData

# 调用深度模型获取识别数据
# /getDeepModelData

# 调用图谱模型访问数据库获取答案
# /getKnowledgeModelData


@mainController_Blueprint.route("/getAnswer",methods=['get','post'])
def getAnswer():
    
    startTime = time.time()
    
    json_data = request.get_json()
    
    
    
    
    question = {
        "question":json_data['question']
                }
    
    # question = {
    #     "question":'aaa'
    #             }
    
    
    resultView = []
    errorView = []
    
    
    coupleResult , errorCouple = doCoupleModel(question)
    
    # 若问题在数据库中匹配到答案，则直接返回简单答案
    if coupleResult['have'] == True:
            
        resultView.append(coupleResult)
    else:
        
        # 问答对模型的错误信息
        if errorCouple['isError'] == True:
                    
            errorView.append(errorCouple)
    
        # 调用深度模型获取识别信息
        deepResult , errorDeep = doDeepModel(question)
    
        # 若深度模型不出错，则进入下一流程
        if errorDeep['isError'] == False:
            # 调用图谱模型查询
            knowledgeResult , errorKnowledge = doKnowledgeModel(deepResult)
            
            if errorKnowledge['isError'] == False:
                    
                resultView.append(knowledgeResult)
            else:
                    
                errorView.append(errorKnowledge)
        else:
            errorView.append(errorDeep)
            

    # 这里返回数据
    
    showJson = {
            "view":resultView,
            "error":errorView,
        }
    
    current_app.logger.warning("getAnswer time: " + str(time.time() - startTime))
    
    return jsonify(showJson)
    
# 调用问答对模型，存储数据
@mainController_Blueprint.route( cfgG.Controller_preURL + "/saveQuestionCouple",methods=['get','post'])
def saveQuestionCouple():
    
    json_data = request.get_json()
    
    coupleData = {
        "question":json_data['question'],
        "answer":json_data['answer']
                }
    
    # 调用问答对模型存储数据
    r = requests.get( cfgG.HOST_PORT + cfgG.CoupleModel_preURL + "/saveQuestionCouple",json=coupleData )
    

    return 


# 调用问答对模型获取数据
def doCoupleModel(questionJSON):
    
    # 需要返回给交互界面的数据
    resultJSON = {
        "id": 2,
        "name": "问答对模型视图",
        "question":"",
        "answer": "",
        "have":False
    }
    
    # 调用模型获取的数据    
    r = coupleMC.doCoupleModel(questionJSON)
    
    rJson = json.loads(r)
    
    view = rJson['view']
    
    error = rJson['error']
    
    resultJSON['question'] = questionJSON['question']
    resultJSON['answer'] = view['answer']
    resultJSON['have'] = view['have']
    
    return resultJSON,error


# 调用深度模型获取数据
def doDeepModel(questionJSON):
    
    # 需要返回给交互界面的数据
    resultJSON = {
        "id": 1,
        "name" : "深度模型视图",
        "origin" : "",
        "object" : "",
        "subject": "",
        "relation": "", 
    }
    
    
    # 调用模型获取的数据
    # r = requests.get( cfgG.HOST_PORT + cfgG.DeepModel_preURL + "/getDeepModelData",json=questionJSON )
    
    r = deepMC.doDeepModel(questionJSON)
    
    rJson = json.loads(r)
    
    view = rJson['view']
    
    error = rJson['error']
    
    resultJSON['origin'] = questionJSON['question']
    resultJSON['object'] = view['object']
    resultJSON['subject'] = view['subject']
    resultJSON['relation'] = view['relation']
    
    
    return resultJSON , error

# 调用知识图谱模型查询数据
def doKnowledgeModel(deepResultJSON):
    
    resultJSON = {
        "id": 3,
        "name": "知识图谱模型视图",
        "CQL": "",
        "return": "",
        "answer":"",
        "question":deepResultJSON['origin']
    }
    
    # 调用模型获取的数据
    # r = requests.get( cfgG.HOST_PORT + cfgG.KnowledgeModel_preURL + "/getKnowledgeModelData",json=deepResultJSON )
    
    r = knowledgeMC.doKnowledgeModel(deepResultJSON)
    
    rJson = json.loads(r)
    
    view = rJson['view']
    
    error = rJson['error']
    
    resultJSON['CQL'] = view['CQL']
    resultJSON['return'] = view['return']
    resultJSON['answer'] = view['answer']
    
    
    return resultJSON,error



   












