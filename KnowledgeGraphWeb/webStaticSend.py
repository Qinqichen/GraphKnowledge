# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 16:11:39 2021

@author: QinQichen
"""

import flask
from flask import request,jsonify 


# 实例化 flask
app = flask.Flask(__name__)

@app.route("/")
def index():
    pass

    return app.send_static_file('index.html')




if __name__ == '__main__':
    app.run(port=1234, debug=True)