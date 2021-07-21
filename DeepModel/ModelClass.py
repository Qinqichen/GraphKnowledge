# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 14:20:57 2021

@author: 秦琦琛

@pythonVersion: python-3.8
"""

"""
模型类文件

1.实体识别
2.关系检测

"""

class ModelClass:
    
    
    def __init__(self):
        # 构造函数
        
        # print('函数构造成功')
        pass
        
    
    def doModelFunction(self,question,num = 1):
        
        result = [];
        
        if num == 1:
            result = self.EntityIdentify(question)
        elif num == 2 :
            result  = self.RelationIdentify(question)
        else:
            result =[ {
                     'id':1,
                     'isError':True,
                     'des':'ModeClass.doModelFunction使用了非法参数 ' + str(num)
                    }]
        # result = {
        #             'enity':enity,
        #             'relation':relation,
        #             'error':error
        #         }
        return result 
    
    
    def EntityIdentify(self,question):
        
        
        # 实体识别函数，调用功能获取结果
        
        entity = [
                {
                    'enity':'青花瓷',
                    'type':'object'
                },
                {
                    'enity':'周星驰',
                    'type':'subject'
                }
            ]
        
        return entity



    def RelationIdentify(self,question):
        
        
        # 实体识别函数，调用功能获取结果
        
        relation = [
                {
                    'relation':'周星驰与青花瓷的关系',
                    'description':'关系'
                }
            ]
        
        return relation














