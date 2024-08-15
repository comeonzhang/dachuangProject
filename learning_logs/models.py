from django.db import models
from django.contrib.auth.models import User
# Create your models here.


class Infor(models.Model):
    owner = models.ForeignKey(User, on_delete=models.CASCADE)  # 外键，关联到User模型，并且是主键
    date_added = models.DateTimeField(auto_now=True)

    eat_times = models.IntegerField()
    eat_begin_time = models.CharField(max_length=20)
    eat_end_time = models.CharField(max_length=20)
    sleep_times = models.IntegerField()
    sleep_begin_time = models.CharField(max_length=20)
    sleep_end_time = models.CharField(max_length=20)

    def __str__(self):
        return f"{self.owner}'s eating and sleeping info at {self.date_added}"

class SleepRecord(models.Model):
    owner = models.ForeignKey(User, on_delete=models.CASCADE)
    date_added = models.DateField(auto_now=True)

    sleep_times = models.IntegerField()
    sleep_begin_time = models.CharField(max_length=20)
    sleep_end_time = models.CharField(max_length=20)

    def __str__(self):
        return f"{self.owner}'s sleeping info at {self.date_added}"

class EatRecord(models.Model):
    owner = models.ForeignKey(User, on_delete=models.CASCADE)
    date_added = models.DateField(auto_now=True)

    eat_times = models.IntegerField()
    eat_begin_time = models.CharField(max_length=20)
    eat_end_time = models.CharField(max_length=20)

    def __str__(self):
        return f"{self.owner}'s eating info at {self.date_added}"
