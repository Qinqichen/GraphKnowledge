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
        
        result = []
        error = {}
        
        if num == 1:
            result,error = self.EntityIdentify(question)
        elif num == 2 :
            result,error  = self.RelationIdentify(question)
        else:
            error = {
                     'id':1,
                     'isError':True,
                     'des':'ModeClass.doModelFunction使用了非法参数 ' + str(num)
                    }
            
            
        return result , error
    
    
    def EntityIdentify(self,question):
        
        
        # 实体识别函数，调用功能获取结果
        
        entity = [
                {
                    'entity':'青花瓷',
                    'type':'object'
                },
                {
                    'entity':'周星驰',
                    'type':'subject'
                }
            ]
        
        error = {
                'id':-1,
                'isError':False,
                'type':'',
                'description':'',
                'model':'DeepModel EntityIdentify'
            }
        
        return entity, error



    def RelationIdentify(self,question):
        
        
        # 实体识别函数，调用功能获取结果
        
        relation = [
                {
                    'relation':'周星驰与青花瓷的关系',
                    'description':'关系'
                }
            ]
        
        error = {
                'id':-1,
                'isError':False,
                'type':'',
                'description':'',
                'model':'DeepModel RelationIdentify'
            }
        
        return relation,error














