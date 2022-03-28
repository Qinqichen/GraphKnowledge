# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 16:22:58 2021

@author: QinQichen
"""

from flask import Flask, url_for, request, redirect, render_template
from flask_sqlalchemy import SQLAlchemy
import CoupleModel.config as config
import logging
import ConfigKnowledge as cfgG



app = Flask(__name__)



app.config.from_object(config)



handler = logging.FileHandler(cfgG.LOG_FILE_PATH)

app.logger.addHandler(handler)

db = SQLAlchemy(app)



