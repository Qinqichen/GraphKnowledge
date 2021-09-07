
from flask import request,Blueprint
import json
from .Neo4jDBController import Neo4jControllerClass

knowledgeModel_Blueprint = Blueprint("knowledgeModel", __name__)



scheme = "neo4j"  # Connecting to Aura, use the "neo4j+s" URI scheme
host_name = "127.0.0.1"
port = 7687
url = "{scheme}://{host_name}:{port}".format(scheme=scheme, host_name=host_name, port=port)
user = "neo4j"
password = "qinqichen"
neo4jClass = Neo4jControllerClass(url, user, password)



# 将预测函数定义为一个端点
@knowledgeModel_Blueprint.route("/getKnowledgeModelData", methods=["GET","POST"])
def KnowledgeModel():
    json_data = request.get_json()
    
    '''
    {#图谱查询模型请求参数
        "id": 1,
        "name" : "深度模型视图",
        "origin" : "青花缠枝牡丹纹罐是什么朝代得？",
        "object" : "青花缠枝牡丹纹罐 ",
        "subject": "朝代",
        "relation": "是什么时间",
    }
    '''
    
    return doKnowledgeModel(json_data)

def doKnowledgeModel(json_data):
    

    '''
    {#图谱查询模型请求参数
        "id": 1,
        "name" : "深度模型视图",
        "origin" : "青花缠枝牡丹纹罐是什么朝代得？",
        "object" : "青花缠枝牡丹纹罐 ",
        "subject": "朝代",
        "relation": "是什么时间",
    }
    '''
    data = { #需返回数据
        "id": 3,
        "name": "知识图谱模型视图",
        "CQL": "MATCH (n:object)-[r1:object_time]->(m:time)",
        "return": "n: 青花缠枝牡丹纹罐, m:元朝",
        "answer":"知识图谱模型答案", #新加字段，存放最终答案
    }
    error = {"id":-1,
             "type":"",
             "description":"",
             "model":"图谱查询模型",
             "isError":False
             }
    
    # # 测试语句
    # if json_data['origin'] == '图谱模型出错' :
        
    #     error['isError'] = True
    
    
    
    showJson = {
        "view": data,
        "error": error,
    }

    return json.dumps(showJson)

@knowledgeModel_Blueprint.route("/getIndexShowGraphData", methods=["GET","POST"])
def getIndexShowGraphData():
    
    
    
    showData = neo4jClass.getIndexShowData()
    
    
    return json.dumps(showData)
