from django.db import models


class Video(models.Model):
    file = models.FileField(upload_to='videos/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return str(self.file.name)
