from django import forms


class UploadForm(forms.Form):
    video = forms.FileField(label='Виберіть відео MP4.')

    def clean_video(self):
        f = self.cleaned_data['video']
        if f.content_type not in ('video/mp4', 'video/x-matroska', 'application/octet-stream'):
            if not f.name.lower().endswith('.mp4'):
                raise forms.ValidationError('Please upload an MP4 file.')
        return f
