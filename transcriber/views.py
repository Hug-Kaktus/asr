import os
import subprocess
import uuid
from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings
from .forms import UploadForm
from .models import Video
from tensorflow import keras
from main import CTCLossModel

# midsize_wav2vec = keras.models.load_model('./midsize_wav2vec.keras', custom_objects={"CTCLossModel": CTCLossModel})


def transcribe_with_midsize_wav2vec(audio_path: str) -> str:
    """
    pretend transcription function. Replace with your actual model inference.
    """
    # text = transcribe_using_neural_network(midsize_wav2vec, audio_path)
    with open('./transcriber/static/texts/videoplayback.txt', 'r', encoding="utf-8") as f:
        text = f.read()
    return text


def extract_audio_from_video(video_path: str, out_wav_path: str) -> None:
    """
    Uses ffmpeg to extract audio to a WAV file with 16 kHz mono PCM — typical for many ASR models.
    Requires `ffmpeg` on PATH.
    """
    os.makedirs(os.path.dirname(out_wav_path), exist_ok=True)

    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-ac', '1',        # mono
        '-ar', '16000',    # 16 kHz
        '-vn',             # no video
        out_wav_path
    ]
    completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {completed.stderr.decode('utf-8', errors='ignore')}")


def upload_view(request):
    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            f = form.cleaned_data['video']
            video = Video.objects.create()
            # save file to Video.file
            video.file.save(f.name, f)
            video.save()

            out_audio_name = f"{uuid.uuid4().hex}.wav"
            out_audio_path = os.path.join(settings.MEDIA_ROOT, 'audio', out_audio_name)

            try:
                extract_audio_from_video(video.file.path, out_audio_path)
            except Exception as e:
                video.file.delete(save=False)
                video.delete()
                return render(request, 'transcriber/upload.html', {
                    'form': form,
                    'error': f'Failed to extract audio: {e}'
                })

            transcript = transcribe_with_midsize_wav2vec(out_audio_path)

            # store transcript in session
            request.session['transcript'] = transcript
            request.session['video_pk'] = video.pk

            return redirect('transcriber:result', pk=video.pk)
    else:
        form = UploadForm()

    return render(request, 'transcriber/upload.html', {'form': form})


def result_view(request, pk):
    video = get_object_or_404(Video, pk=pk)
    transcript = request.session.get('transcript')
    return render(request, 'transcriber/result.html', {
        'video': video,
        'transcript': transcript,
    })
