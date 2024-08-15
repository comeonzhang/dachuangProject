"""为应用程序users定义URL模式"""

from django.urls import path, include

from . import views

app_name = 'users'
urlpatterns = [
    #包含默认身份验证URL（登陆页面与http://localhost:8000/users/login/匹配）
    path('', include('django.contrib.auth.urls')),#，users在users/urls.py查找，login让Django发送请求给默认视图login
    #注册页面
    path('register/', views.register, name='register')
]