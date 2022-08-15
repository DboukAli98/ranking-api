from flask import Flask, request, redirect, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import textdistance as td
from sklearn.feature_extraction.text import TfidfVectorizer
from pdfminer.high_level import extract_text

# def RankCandidate(Resume , Job_Desc):
#     text = [Resume,Job_Desc]
#     cv = CountVectorizer()
#     count_matrix = cv.fit_transform(text)
#     cosine_similarity(count_matrix)
#     matchpercentage = cosine_similarity(count_matrix)[0][1]
#     matchpercentage = round(matchpercentage*100,2)
#     return print('Candidate Resume match {}% '.format(matchpercentage) + 'your Job Requirements')





#tokenizer
def do_tfidf(token):
    tfidf = TfidfVectorizer(max_df=1.0, min_df=1)
    words = tfidf.fit_transform(token)
    sentence = " ".join(tfidf.get_feature_names())
    return sentence




#different similarity algorithms
def match(resume, job_des):
    j = td.jaccard.similarity(resume, job_des)
    s = td.sorensen_dice.similarity(resume, job_des)
    c = td.cosine.similarity(resume, job_des)
    o = td.overlap.normalized_similarity(resume, job_des)
    total = (j+s+c+o)/4
    match_percentage = round(total*100.2)
    # total = (s+o)/2
    return "Candidate's Profile match {}%" .format(match_percentage) +" of your job requirements"


    


app = Flask(__name__)

@app.route('/home')
def Home():
    return "Hello World"

@app.route('/rank-candidate' , methods=['POST'])
def Rank():
    json_data = request.get_json(force=True)
    resume = json_data["resume"]
    job_desc = json_data["job_desc"]
    return jsonify(match(do_tfidf([resume]), do_tfidf([job_desc])))
    

if __name__ == "__main__":
    app.run()

