"""定义learning_logs的URL模式(区分)"""

from django.urls import path    #来映射到视图

from . import views #'.'当前导入

app_name = 'learning_logs'  #与其他同名文件区分开  《命名空间》
urlpatterns = [
    #主页
    path('', views.index, name='index'), #搜索所有来匹配，‘’与基础URL匹配，空则忽略基础；1参匹配则调用2参函数；3参直接使用名称，不用编写URL
    path('infors/', views.Information, name='infors'),#/可省
    path('tcp_server/', views.tcp_server, name='tcp_server'),
    path('api/receive_data/', views.receive_data, name='receive_data'),
]
