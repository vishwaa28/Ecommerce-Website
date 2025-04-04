from app import app
from flask import Flask, render_template, request, redirect, url_for, jsonify, flash, session

import torch
from flask_sqlalchemy import SQLAlchemy


if __name__=="__main__":
    app.run(debug=True)