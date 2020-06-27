#!/usr/bin/python
# -*- coding: utf-8 -*-

import re
import requests
import timeit
import math
import numpy
import operator
from bs4 import BeautifulSoup
from elasticsearch import Elasticsearch
from flask import request
from flask_restful import Resource
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
    pattern = '[(©).,0-9]'
    text = re.sub(pattern=pattern, repl='', string=text)
    text = text.lower().replace('[', '').replace(']', '').replace("'", "").replace('\n', '') \
        .replace('"', '').replace('-', '').replace('\t', '').replace('?', '').replace('@', '').replace('#', '').split()
    return text


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
    sent_list_1.append(s)
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
    # build set of words
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


class UrlReceived(Resource):  # 단일 url 입력시
    def post(self):
        try:
            url = Resource
            url = url.repalce("\n", "")
            start_time = []
            stop_time = []
            result_time = []
            tmp = 0
            start_time.append(timeit.default_timer())
            res = requests.get(url)
            html = BeautifulSoup(res.content, "html.parser")
            html_body = html.find_all('div')
            dictionary = {}
            for string in html_body:
                word = str(string.text.split())
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
                word_doc.append(key)
            stop_time.append(timeit.default_timer())
            result_time.append(stop_time[0] - start_time[0])  # 총 소요시간 result_time[0]
            result = [word_doc, tmp, result_time[0], url]  # [0]중복없는 단어 리스트, [1]총 단어수(중복X), [2]소요시간, [3]url
            return result
        except:
            print("Can not open this URL")


class FileReceived(Resource):  # 다중 url 텍스트 입력 (Resource : 파일 이름)
    def post(self):
        whole_word_info = []  # 각 url 총단어 저장
        url = []
        status = []
        word_num = []  # 각 url 단어수 저장
        start_time = []
        stop_time = []
        result_time = []
        overlap_url = []  # url 중복 체크리스트
        url_index = 0
        try:
            f = request.files[Resource]
            url_lines = f.readlines()
            f.close()
            for i in url_lines:
                url.append(i)
                url[url_index] = url[url_index].replace("\n", "")
                try:
                    res = requests.get(url[url_index])
                    tmp = 0
                    start_time.append(timeit.default_timer())
                    html = BeautifulSoup(res.content, "html.parser")
                    html_body = html.find_all('div')
                    dictionary = {}
                    for string in html_body:
                        word = str(string.text.split())
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
                        word_doc.append(key)
                    stop_time.append(timeit.default_timer())
                    whole_word_info[url_index] = [word_doc]  # 중복없는 단어 저장
                    word_num[url_index] = tmp  # 단어 수 저장
                    result_time.append(stop_time[url_index] - start_time[url_index])
                    status.append("O")
                    url_index += 1
                except:  # url 읽어오지 못할시 오류처리
                    status.append("X")
                    result_time.append("-")
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
                    url.pop(k)
                    status.pop(k)
                    overlap_url.pop(k)
                    b += 1
            url_index -= b
            result = [url, overlap_url, status, result_time, whole_word_info, word_num]
            # url, 중복여부, 크롤링성공여부, 소요시간, 각 url 마다의 단어, 각 url 마다의 단어수
            return result
        except:  # 파일읽기 시 오류처리
            print("File Read Error!")


class WordAnalysis(Resource):  # tf-idf 단어 top 10
    def post(self):
        global es
        global sent_list_2
        es.list = []
        es_list = es.search('multi')
        n = 0
        words = []
        url = []
        tf_idf_word = []
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
        for word, tf_val in tf_d.iteritems():
            dic[word] = tf_val * idf_d[word]
        dic = sorted(dic.items(), key=operator.itemgetter(1, 0), reverse=True)
        for i in range(0, 10):
            word[i] = str(dic[i][0])
            tf_idf_word[i] = str(dic[i][1])
        result = [word, tf_idf_word]  # top10 단어와 유사도 (리스트 10개씩) 리턴
        return result


class SimAnalysis(Resource):  # cosine 유사성 분석시
    def post(self):
        global es
        es.list = []
        es_list = es.search('multi')
        n = 0
        words = []
        url = []
        key = []
        value = []
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
        for i in range(0, 3):
            key[i] = str(cos_simil[i][0])
            value[i] = str(cos_simil[i][1])
        result = [key, value]
        return result


if __name__ == '__main__':
    es = Elasticsearch([{'host': "127.0.0.1", 'port': "9200"}], timeout=30)
