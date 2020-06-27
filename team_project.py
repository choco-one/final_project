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
    cleaned_text = re.sub('[-=+©,#/\?:;\{\}^$.@—{*\"※;»~&}%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', text).replace("’",
                                                                                                         "").split()
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


class UrlReceived(Resource):  # 단일 url 입력시
    def post(self, input_url):
        url = input_url
        url = url.replace("\n", "")
        start_time = []
        stop_time = []
        result_time = []
        tmp = 0
        try:
            res = requests.get(url)
        except:
            print("Can not open this URL")
        start_time.append(timeit.default_timer())
        html = BeautifulSoup(res.content, "html.parser")
        html_body = html.find_all('div')
        dictionary = {}
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
        result = [str(tmp), str(result_time[0]), url]  # [0]중복없는 단어 리스트, [1]총 단어수(중복X), [2]소요시간, [3]url
        print(result)
        return result


class FileReceived(Resource):  # 다중 url 텍스트 입력 (url_txt : 파일 이름)
    def post(self, url_txt):
        # print(1)
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
            f = open(url_txt, 'r')
            url_lines = f.readlines()
            f.close()
        except:  # 파일읽기 시 오류처리
            print("File Read Error!")
        for i in url_lines:
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
                url.pop(k)
                status.pop(k)
                overlap_url.pop(k)
                result_time.pop(k)
                whole_word_info.pop(k)
                word_num.pop(k)
                b += 1
        url_index -= b
        result = [url, overlap_url, status, result_time, whole_word_info, word_num]
        # url, 중복여부, 크롤링성공여부, 소요시간, 각 url 마다의 단어, 각 url 마다의 단어수
        return result


class WordAnalysis(Resource):  # tf-idf 단어 top 10
    def post(self):
        global es
        global sent_list_2
        es.list = []
        es_list = es.search(index='multi')
        # print(2)
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
        for i in range(0, 10):
            result_word.append(dic[i][0])
            tf_idf_word.append(dic[i][1])
        result = [result_word, tf_idf_word]  # top10 단어와 유사도 (리스트 10개씩) 리턴
        print(result)
        return result


class SimAnalysis(Resource):  # cosine 유사성 분석시
    def post(self):
        global es
        es.list = []
        es_list = es.search(index='multi')
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
            key.append(cos_simil[i][0])
            value.append(cos_simil[i][1])
        result = [key, value]
        print(result)
        return result


resultForSearchAPI = []


def searchAPI():
    global es
    body = {'query': {
        "match_all: {}"
    }}
    res1 = es.search(index='single', body=body)
    res2 = es.search(index='multi', body=body)
    # resultForSearchAPI에 전부 저장됨
    del resultForSearchAPI[:]
    resultForSearchAPI.append(res1)
    resultForSearchAPI.append(res2)


# res의 저장을 단일과 중복url을 분리할까 말까 고민하고 있다.
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


# URL이 있으면 TRUE로 없다면 Flase
def isthereurl():
    if url_id > 1:
        return True
    if _id > 1:
        return True
    else:
        return False


if __name__ == '__main__':
    es = Elasticsearch([{'host': "127.0.0.1", 'port': "9200"}], timeout=30)
    # 예시
    res_craw_url = UrlReceived().post("http://wicket.apache.org\n")       # url이름 입력
    save1(res_craw_url[0], res_craw_url[1], res_craw_url[2])
    result_crawling_url = FileReceived().post("url.txt")                   # 파일이름 입력
    save2(result_crawling_url[4], result_crawling_url[5], result_crawling_url[2], result_crawling_url[3],
          result_crawling_url[0])

    # tf-idf 단어 top 10
    word_sim_result = WordAnalysis().post()
    word_sim_word = word_sim_result[0]  # word 와 figure에 10개씩의 인덱스 존재
    word_sim_figure = word_sim_result[1]

    # cosine 유사도 분석
    cos_sim_result = SimAnalysis().post()  # url 과 figure에 3개씩의 인덱스 존재
    cos_sim_url = cos_sim_result[0]
    cos_sim_figure = cos_sim_result[1]
