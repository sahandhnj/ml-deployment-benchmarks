from locust import HttpLocust, TaskSet, task


class UserTasks(TaskSet):

    @task
    def predict(self):
        with open('input.jpg', 'rb') as image:
            self.client.post(
                "/predict",
                data={},
                files={'file': image}
            )

class WebsiteUser(HttpLocust):
    task_set = UserTasks