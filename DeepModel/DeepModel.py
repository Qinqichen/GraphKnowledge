import flask
import pandas as pd
import torch
from flask import request,jsonify

# 实例化 flask
app = flask.Flask(__name__)

# 加载模型
#model = torch.load('modelname')

# 将预测函数定义为一个端点
@app.route("/getDeepModelData", methods=["GET","POST"])

def DeepModel():
    json_data = request.get_json()
    print(json_data)

    '''
    {#深度模型请求参数
        "question":"青花缠枝牡丹纹罐是什么朝代得？"
                }
    '''

    data = {
        "id": 1,
        "name" : "深度模型视图",
        "origin" : "青花缠枝牡丹纹罐是什么朝代得？",
        "object" : "青花缠枝牡丹纹罐 ",
        "subject": "朝代",
        "relation": "是什么时间",
    }
    error = {"id":1,
             "type":"timeOut",
             "description":"模型超时",
             "model":"深度模型",
             "isError":False
             }
    # 测试语句
    if json_data['question'] == '深度模型出错' :
        
        error['isError'] = True
    
    
    showJson = {
        "view": data,
        "error": error,
    }

    return jsonify(showJson)
if __name__ == '__main__':

    app.run(port=1235, debug=True)
