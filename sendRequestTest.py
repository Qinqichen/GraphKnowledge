# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 20:58:02 2021

@author: 秦琦琛

@pythonVersion: python-3.8
"""


import requests;
import json;

returnJson = [
    {#需返回数据
        "id": 1,
        "name" : "深度模型视图",
        "origin" : "青花缠枝牡丹纹罐是什么朝代得？",
        "object" : "青花缠枝牡丹纹罐 ",
        "subject": "朝代",
        "relation": "是什么时间"
    },
    {#需返回数据
        "id": 2,
        "name": "问答对模型视图",
        "question": "青花缠枝牡丹纹罐是什么朝代得？",
        "answer": "",
        "have":False,#新加字段，判断是否存在答案
    },
    { #需返回数据
        "id": 3,
        "name": "知识图谱模型视图",
        "CQL": "MATCH (n:object)-[r1:object_time]->(m:time)",
        "return": "n: 青花缠枝牡丹纹罐, m:元朝",
        "answer":"知识图谱模型答案", #新加字段，存放最终答案
    }
]



sendJSON = [{#深度模型请求参数
        "question":"青花缠枝牡丹纹罐是什么朝代得？"
                },
    {#问答对模型请求参数
        "question":"青花缠枝牡丹纹罐是什么朝代得？"
                },
    {#图谱查询模型请求参数
        "id": 1,
        "name" : "深度模型视图",
        "origin" : "青花缠枝牡丹纹罐是什么朝代得？",
        "object" : "青花缠枝牡丹纹罐 ",
        "subject": "朝代",
        "relation": "是什么时间",
    }
]





def getResultByPost(question):
 
    questionJson = {
            "question":question
        }
    
    # 获取数据
    r = requests.get("http://127.0.0.1:1234/knowledgeGraphServer/getAnswer",json=questionJson)
    # r = requests.get("http://localhost:1234/getAnswer")


    print(r)
    
    return r.json()

if __name__ == '__main__':
    
    # result = getResultByPost("交互界面请求的问题" )

    
    # for error in result['error']:
        
    #     print(error)
            
    
    # print(result['view'])
    result = getResultByPost("交互界面请求的问题" )

    
    print(result)















