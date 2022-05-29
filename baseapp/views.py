from django.shortcuts import render
from django.http import HttpResponse
import joblib

# Create your views here.
'''
def welcome(request):
    return HttpResponse("<h1>Hello</h1>")
'''

def welcome(request):
    return render(request,"index.html")

def user(request):
    username = request.GET['username']
    return render(request, "user.html",{'name': username})

def home(request):
    return render(request,"home.html")

def result(request):

    cls = joblib.load('finalized_model.sav')

    lis = []

    lis.append(request.GET['RI'])
    lis.append(request.GET['Na'])
    lis.append(request.GET['Mg'])
    lis.append(request.GET['Al'])
    lis.append(request.GET['Si'])
    lis.append(request.GET['K'])
    lis.append(request.GET['Ca'])
    lis.append(request.GET['Ba'])
    lis.append(request.GET['Fe'])

    # print(lis)
    ans = cls.predict([lis])

    return render(request,"result.html",{'ans':ans, 'lis':lis})