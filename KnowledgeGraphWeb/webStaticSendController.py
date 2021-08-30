# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 16:11:39 2021

@author: QinQichen
"""

import flask
from flask import request,jsonify , Blueprint


webStaticController_Blueprint = Blueprint("webStatic", __name__,static_folder='static')



@webStaticController_Blueprint.route("/")
def index():
    pass

    return webStaticController_Blueprint.send_static_file('index.html')




# if __name__ == '__main__':
#     app.run(port=1234, debug=True)