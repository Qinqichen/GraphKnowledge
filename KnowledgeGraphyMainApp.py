# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 19:13:54 2021

@author: 秦琦琛

@pythonVersion: python-3.8
"""

import ConfigKnowledge as cfgG

from init import app


from MainController import mainController_Blueprint
from CoupleModel.CoupleModelController import coupleModel_Blueprint
from DeepModel.DeepModel import deepModelController_Blueprint
from KnowledgeModel.KnowledgeModelController import knowledgeModel_Blueprint
 
from CoupleModel.QuestionDB import questionDB_Blueprint

from KnowledgeGraphWeb.webStaticSendController import webStaticController_Blueprint


app.register_blueprint(mainController_Blueprint, url_prefix=cfgG.Controller_preURL)
app.register_blueprint(coupleModel_Blueprint,url_prefix=cfgG.CoupleModel_preURL)
app.register_blueprint(webStaticController_Blueprint,url_prefix=cfgG.Web_preURL)
app.register_blueprint(deepModelController_Blueprint,url_prefix=cfgG.DeepModel_preURL)
app.register_blueprint(knowledgeModel_Blueprint,url_prefix=cfgG.KnowledgeModel_preURL)
app.register_blueprint(questionDB_Blueprint,url_prefix=cfgG.QuestionDB_preURL)


if __name__ == '__main__':
    
    app.run(port=1234, debug=True)
    





