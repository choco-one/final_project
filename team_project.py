#!/usr/bin/python
# -*- coding: utf-8 -*-

import re
import requests
import timeit
import math
import numpy
import operator
from bs4 import BeautifulSoup
from flask import Flask, request
from elasticsearch import Elasticsearch
from nltk.tokenize import word_tokenize

# 입력방법 2가지 -> 단일 url 입력(유사도 분석 X, 단어 분석은 가능?), 다중 url 텍스트파일 입력(유사도 분석 O)
# 사이트에서 크롤링 -> URL 별 전체 단어수 구하기(처리시간 같이)
# TF-IDF 기반 함수 정의 -> 상위 top10 주요 단어 리스트 생성 (단어 분석 버튼)
# cosine similarity 기반 현재 url 과 가장 유사한 top3 url 리스트 (유사도 분석 버튼)
# url 단어수 출력, 주요단어 top 10, 유사도 분석 기능


word_d = {}
sent_list_1 = []
sent_list_2 = []


def cleansing(text):  # 특수문자 제거 함수
    cleaned_text = re.sub('[-=+©,#/\?:;\{\}^$.@—{*\"※;»~&}%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', text).replace("’", "").split()
    return cleaned_text


def make_vector(i):
    global sent_list_1
    v = []
    s = sent_list_1[i]
    tokenized = word_tokenize(s)
    for w in word_d.keys():
        val = 0
        for t in tokenized:
            if t == w:
                val += 1
        v.append(val)
    return v


def process_new_sentence_1(s):
    global sent_list_1
    sent_list_1.append(s)
    tokenized = word_tokenize(s)
    for word in tokenized:
        if word not in word_d.keys():
            word_d[word] = 0
        word_d[word] += 1


def process_new_sentence_2(s):
    global sent_list_2
    sent_list_2.append(s)
    tokenized = word_tokenize(s)
    for word in tokenized:
        if word not in word_d.keys():
            word_d[word] = 0
        word_d[word] += 1


def compute_tf(s):
    bow = set()
    wordcount_d = {}
    tokenized = word_tokenize(s)
    for tok in tokenized:
        if tok not in wordcount_d.keys():
            wordcount_d[tok] = 0
        wordcount_d[tok] += 1
        bow.add(tok)
    tf_d = {}
    for t in bow:
        tf_d[t] = wordcount_d[t] / float(sum(wordcount_d.values()))
    return tf_d


def compute_idf():
    global sent_list_2
    Dval = len(sent_list_2)
    bow = set()
    for i in range(0, len(sent_list_2)):
        tokenized = word_tokenize(sent_list_2[i])
        for tok in tokenized:
            bow.add(tok)
    idf_d = {}
    for t in bow:
        cnt = 0
        for s in sent_list_2:
            if t in word_tokenize(s):
                cnt += 1
        idf_d[t] = math.log10(Dval / float(cnt))
    return idf_d


app = Flask(__name__)


@app.route('/', methods=['GET'])
def homepage():
    upload = "<form action = \"/file_status\" method = \"post\" enctype =\"multipart/form-data\"><input type = " \
             "\"file\" name = \"file\" /><input type = \"submit\" /></form> "

    return "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"UTF-8\"><title>Home</title></head><body><form " \
           "action=\"/url_status\" method=\"post\"><p>Input URL<input type=\"text\" name=\"URL\"></p><input " \
           "type=\"submit\" value=\"Analyze\"></form><p>" + upload + "</p></body></html> "


@app.route('/url_status', methods=['POST'])
def url_status():
    if request.method == "POST":
        url = []
        url.append(request.form['URL'])
        url[0] = url[0].replace("\n", "")
        start_time = []
        stop_time = []
        result_time = []
        dictionary = {}
        status = ""
        tmp = 0
        try:
            res = requests.get(url[0])
            start_time.append(timeit.default_timer())
            html = BeautifulSoup(res.content, "html.parser")
            html_body = html.find_all('div')
            for string in html_body:
                word = str(string.text)
                word = cleansing(word)
                for element in word:
                    if element in dictionary.keys():  # 총 단어수 tmp (중복 허용)/ 단어 리스트는 중복 없이
                        dictionary[element] += 1
                        tmp += 1
                    else:
                        dictionary[element] = 1
                        tmp += 1
            word_doc = []  # 딕셔너리에서 리스트로 단어 옮기기
            for key in dictionary:
                word_doc.append(cleansing(key))
            stop_time.append(timeit.default_timer())
            result_time.append(stop_time[0] - start_time[0])  # 총 소요시간 result_time[0]
            status = "성공"
        except:
            status = "실패"
            tmp = 0
            result_time.append("-")

        e = ""
        e = e + "<tr><td>" + url[0] + "</td><td>" + status + "</td><td>" + str(tmp) + "</td><td>" + str(result_time[0]) + "</td></tr>"

        return "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"UTF-8\"><title>Status text Box</title></head><body><p>Status Text Box</p><table border=1 style=\"border: 1px solid black; text-align:center;\"><th>URL</th><th>Status</th><th>Word Num</th><th>Time</th>" + e + "</table><p><a href=\"/\" target=\"\"><input type =\"submit\" value = \"Go Home\"></a></form></p></body></html>"


@app.route('/file_status', methods=['POST'])
def file_status():
    if request.method == "POST":
        whole_word_info = []  # 각 url 총단어 저장
        url = []  # 각 url 저장
        status = []  # 크롤링 성공여부 저장
        word_num = []  # 각 url 단어수 저장
        start_time = []  # 시작 시간
        stop_time = []  # 끝난 시간
        result_time = []  # 결과 시간
        overlap_url = []  # url 중복 체크리스트
        url_index = 0
        try:
            f = request.files['file']
            url_lines = f.readlines()
            f.close()
        except:  # 파일읽기 시 오류처리
            print("File Read Error!")
        for i in url_lines:
            i = i.decode('utf-8')
            url.append(i)
            url[url_index] = url[url_index].replace("\n", "")
            try:
                res = requests.get(url[url_index])
                status.append("성공")
            except:  # url 읽어오지 못할시 오류처리
                status.append("실패")
                start_time.append('-')
                stop_time.append('-')
                result_time.append("-")
                whole_word_info.append("-")
                word_num.append('-')
                url_index += 1
                continue
            tmp = 0
            start_time.append(timeit.default_timer())
            html = BeautifulSoup(res.content, "html.parser")
            html_body = html.find_all('div')
            dictionary = {}
            for string in html_body:
                word = str(string.text)
                word = cleansing(word)
                for element in word:
                    tmp += 1
                    if element in dictionary.keys():  # 총 단어수 tmp (중복 허용)/ 단어 리스트는 중복 없이
                        dictionary[element] += 1
                    else:
                        dictionary[element] = 1
            word_num.append(tmp)  # 단어 수 저장
            word_doc = []  # 딕셔너리에서 리스트로 단어 옮기기
            for key in dictionary:
                word_doc.append(key)
            whole_word_info.append(word_doc)
            stop_time.append(timeit.default_timer())
            result_time.append(float(stop_time[url_index] - start_time[url_index]))
            url_index += 1
        count = len(url)
        for i in range(0, count):
            check = 0
            for j in range(0, count):
                if url[i] == url[j] and i != j:
                    check = 1
                    break
            if check == 1:  # 중복일 경우
                overlap_url.append("O")
            else:
                overlap_url.append("X")
        index = []
        b = 0
        for i in range(0, count):
            index.append(0)

        for i in range(0, url_index):
            if overlap_url[i] == "O":
                ovl = url[i]
                for j in range(i + 1, count):
                    if url[j] == ovl:
                        index[j] = 1
        for k in range(0, url_index):
            if index[k] == 1:
                url.pop(k-b)
                status.pop(k-b)
                overlap_url.pop(k-b)
                result_time.pop(k-b)
                whole_word_info.pop(k-b)
                word_num.pop(k-b)
                b += 1
                url_index -= 1
        status_result = ""
        for i in range(0, url_index):
            status_result = status_result + "<tr><td>" + url[i] + "</td><td>" + status[i] + "</td><td>" + str(word_num[i]) + "</td><td>" + overlap_url[i] + "</td><td>" + str(result_time[i]) + "</td></tr>"
        save2(whole_word_info, word_num, status, result_time, url)  # input data in elastic search

        return "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"UTF-8\"><title>Status text " \
               "Box</title></head><body><p>Status Text Box</p><table border=1 style=\"border: 1px solid black; " \
               "text-align:center;\"><th>URL</th><th>Status</th><th>Word Num</th><th>중복 여부</th><th>Time</th>" + \
               status_result + "</table><p><a href=\"/tf_idf\" target=\"_blank\"><input type=\"submit\" " \
                               "value=\"TF-IDF\"></a></p><p><a href=\"/cosine\" target=\"_blank\"><input " \
                               "type=\"submit\" value=\"Cosine-Similarity\"></a></p><p><a href=\"/\" " \
                               "target=\"\"><input type =\"submit\" value = \"Go Home\"></a></form></p></body></html> "


@app.route('/tf_idf', methods=['GET'])
def tf_idf():
    global es
    global sent_list_2
    es.list = []
    es_list = es.search(index='multi')
    n = 0
    words = []
    url = []
    tf_idf_word = []
    result_word = []
    for i in es_list['hits']['hits']:
        words.append(i['_source']['word'])
        url.append(i['_source']['url'])
        n += 1
    if n < 2:
        return "Need More Documents"
    sentence = []
    for i in range(0, n):
        sentence.append("")
    for i in range(0, n):
        for j in words[i]:
            sentence[i] += j + " "
    for i in range(0, n):
        process_new_sentence_2(sentence[i])
    idf_d = compute_idf()
    for i in range(0, len(sent_list_2)):
        tf_d = compute_tf(sent_list_2[i])
    dic = {}
    for word, tf_val in tf_d.items():
        dic[word] = tf_val * idf_d[word]
    dic = sorted(dic.items(), key=operator.itemgetter(1, 0), reverse=True)
    e = ""
    for i in range(0, 10):
        result_word.append(dic[i][0])
        tf_idf_word.append(dic[i][1])
        e = e + "<tr><td>"+str(result_word[i])+"</td><td>"+str(tf_idf_word[i])+"</td></tr>"
    return "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"UTF-8\"><title>TF-IDF " \
           "Top10</title></head><body><p>TF-IDF Top10</p><table border=1 style=\"border: 1px solid black; " \
           "text-align:center;\"><th>Word</th><th>TF-IDF</th>" + e + "</table></body></html> "


@app.route('/cosine',methods=['GET'])
def cosine():
    global es
    es.list = []
    es_list = es.search(index='multi')
    n = 0
    words = []
    url = []
    for i in es_list['hits']['hits']:
        words.append(i['_source']['word'])
        url.append(i['_source']['url'])
        n += 1
    if n < 2:
        return "Need More Documents"
    sentence = []
    for i in range(0, n):
        sentence.append("")
    for i in range(0, n):
        for j in words[i]:
            sentence[i] += j + " "
    for i in range(0, n):
        process_new_sentence_1(sentence[i])
    dotpro = {}
    cos_simil = {}
    for i in range(0, n):
        for j in range(i + 1, n):
            v_1 = make_vector(i)
            v_2 = make_vector(j)
            index = str(i) + '-' + str(j)
            dotpro[index] = numpy.dot(v_1, v_2)
            cos_simil[index] = dotpro[index] / numpy.linalg.norm(v_1) * numpy.linalg.norm(v_2)
    cos_simil = sorted(cos_simil.items(), key=operator.itemgetter(1, 0), reverse=True)
    e = ""
    for i in range(0, 3):
        key = str(cos_simil[i][0])
        value = str(cos_simil[i][1])
        keys = key.split('-')
        url_to_url = "[ " + url[int(keys[0])] + " ] - [ " + url[int(keys[1])] + " ]"
        e = e + "<tr><td>" + str(i + 1) + "</td><td>" + url_to_url + "</td><td>" + str(value) + "</td></tr>"
    return "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"UTF-8\"><title>Cosine-Similarlity " \
           "Top3</title></head><body><p>Cosine-Similarlity Top3</p><table border=1 style=\"border: 1px solid black; " \
           "text-align:center;\"><th>Rank</th><th>URL</th><th>Cosine-Similarlity</th>" + e + "</table></body></html> "


result1 = []  # result1에는 elasticsearch가
result1_1 = []  # 실제 삽입값이 저장된다
result2 = []
result2_1 = []
_id = 1
url_id = 1


# 불러오는 형태로 구현하기 위해서 함수로 구현 하였다.
def save1(frequency, time, url):  # word, frequency, time은 모두 리스트가 아닌 단일값이다.
    global es

    # 전역변수 i로서 구현하였다.
    global _id

    # index를 임의로 정해야진게 아니라면 이후 수정해야한다.
    # 인덱스를 single이라는 단일 url을 나타내게 함.

    insert_es = {
        "frequency": frequency,
        "urltime": time,
        "url": url
    }
    res = es.index(index='single', id=_id, body=insert_es)
    result1_1.append(insert_es)
    result1.append(res)
    print(res)
    _id += 1


# word, frequency등 매개변수는 리스트이다.
def save2(word, frequency, success, time, url):
    global es
    global url_id
    for i in range(len(word)):
        insert_es = {
            "word": word[i],
            "frequency": frequency[i],
            "success": success[i],
            "url_time": time[i],
            "url": url[i]
        }
        # 우선 doc_type에 중복일 경우 'multi'로 표현하였다.
        res = es.index(index='multi', id=url_id, body=insert_es)
        result2_1.append(insert_es)
        result2.append(res)
        print(res)
        url_id += 1


if __name__ == '__main__':
    es = Elasticsearch([{'host': "127.0.0.1", 'port': "9200"}], timeout=30)
    app.run()
