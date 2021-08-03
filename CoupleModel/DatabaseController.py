# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 15:11:34 2021

@author: 秦琦琛

@pythonVersion: python-3.8
"""
"""
后期可能有多个表，采用 外观模式 
对每个表做接口，提供给上层使用


"""

import json
import requests
import sys
sys.path.append('..')
import ConfigKnowledge as cfgG

class DatabaseController:
    
    def __init__(self):
        
        pass 
    

    
    class QuestionCouple:
        
        def __init__(self):
            pass 
        
        def selectByQuestion(self,questionJson):
            
            r = requests.get(cfgG.HOST_PORT+cfgG.QuestionDB_preURL+'/testSelect', json = questionJson)
        
            return r.json()
        
        def insert(self,question,answer = 'None'):
            
            pass
            

    @classmethod
    def QuestionCoupleDB(cls):
        return DatabaseController.QuestionCouple() 



# 测试用函数
if __name__ == '__main__':
    
    json_data = {
            'question': 'ttt'
        }

    answer = DatabaseController.QuestionCoupleDB().selectByQuestion(json_data)
    
    print('-------------------')
    print(answer)
    pass 





