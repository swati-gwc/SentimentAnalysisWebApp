from django.urls import path
from . import views

urlpatterns = [
  
    path('',views.predictor, name="predictor"),
    path('result',views.form_info, name="result"),
  
   
]