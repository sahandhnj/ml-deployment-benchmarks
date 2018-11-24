from locust import HttpLocust, TaskSet

def index(l):
    l.client.post("/predict")


class Predict(TaskSet):
    tasks = {main: 1}

class WebsiteUser(HttpLocust):
    task_set = Predict
    min_wait = 5000
    max_wait = 9000
