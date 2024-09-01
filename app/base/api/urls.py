# urls.py
from django.urls import path
from .views import AnswerView

urlpatterns = [
    path('api/answer/', AnswerView.as_view(), name='answer'),
]