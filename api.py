from crypt import methods
import os
from urllib import response
from model import match,do_tfidf
from flask import Flask, request, redirect, jsonify
from app import app

@app.route('/rank-candidate' , methods=['POST'])
def Rank():
    json_data = request.get_json(force=True)
    resume = json_data["resume"]
    job_desc = json_data["job_desc"]
    return jsonify(match(do_tfidf([resume]), do_tfidf([job_desc])))
    

if __name__ == "__main__":
    app.run()

