# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 19:13:54 2021

@author: 秦琦琛

@pythonVersion: python-3.8
"""

import ConfigKnowledge as cfgG

from werkzeug.serving import run_simple

from werkzeug.middleware.dispatcher import DispatcherMiddleware

from KnowledgeController import app as controller
from CoupleModel.CoupleModel import app as CoupleModel
from DeepModel.DeepModel import app as DeepModel
from KnowledgeModel.KnowledgeModel import app as KnowledgeModel
 
from CoupleModel.QuestionDB import app as QuestionDB

from KnowledgeGraphWeb.webStaticSend import app as WebIndex

app = DispatcherMiddleware(controller,
                           {
                               cfgG.CoupleModel_preURL:CoupleModel,
                               cfgG.DeepModel_preURL:DeepModel,
                               cfgG.KnowledgeModel_preURL:KnowledgeModel,
                               cfgG.QuestionDB_preURL:QuestionDB,
                               '/web':WebIndex
                            })


if __name__ == '__main__':
    run_simple("localhost", cfgG.PORT_GRAPH, app,
               use_reloader=True, use_debugger=True, use_evalex=True ,threaded=True )







