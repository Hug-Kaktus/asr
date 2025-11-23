from django import forms


class UploadForm(forms.Form):
    video = forms.FileField(label='Виберіть медіа MP3 або MP4.')

    def clean_video(self):
        f = self.cleaned_data['video']

        allowed_content_types = (
            'video/mp4',
            'video/x-matroska',
            'audio/mpeg',
            'application/octet-stream',
        )

        allowed_extensions = ('.mp4', '.mkv', '.mp3')

        # MIME type check
        if f.content_type not in allowed_content_types:
            # Extension check as backup
            if not any(f.name.lower().endswith(ext) for ext in allowed_extensions):
                raise forms.ValidationError(
                    'Будь ласка, завантажте файл у форматі MP4 або MP3.'
                )

        return f
