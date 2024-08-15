from django.contrib import admin
from .models import Infor, SleepRecord, EatRecord  #'.'在admin.py所在目录查找models.py
# Register your models here.

admin.site.register(Infor)
admin.site.register(SleepRecord)
admin.site.register(EatRecord)