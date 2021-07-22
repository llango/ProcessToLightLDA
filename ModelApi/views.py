# -*- coding: utf-8 -*-
# @Time    : 2021/7/21 下午7:52
# @Author  : WuDiDaBinGe
# @FileName: views.py
# @Software: PyCharm
import json

from django.http import HttpResponse, JsonResponse
from django.views.decorators.http import require_http_methods
from ModelApi.LightLdaUtil import extractKeyWordByModel
from django.views.decorators.csrf import csrf_exempt

from ModelApi.LightLdaUtil import preTestData


@csrf_exempt
def extract_docs_key_word(request):
    text_dict = json.loads(request.body, encoding='utf-8')
    result = extractKeyWordByModel.run(text_dict)
    return JsonResponse(result)


# 下面接口用到了可执行文件的推断
@csrf_exempt
def infer_by_lightLDA(request):
    """
    传入文章,返回文章的前五个主题以及相似度矩阵
    """
    text_dict = json.loads(request.body, encoding='utf-8')
    doc_topic, cos_matrix = preTestData.infer_doc_topic(text_dict)
    result = {
        'doc_topic': doc_topic,
        'cos_matrix': cos_matrix
    }
    return JsonResponse(result)
