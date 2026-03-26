# v3 - uses static_ffmpeg Python package for reliable ffmpeg
FROM python:3.12-slim

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Force static_ffmpeg to download its binaries now (at build time)
RUN python -c "import static_ffmpeg; static_ffmpeg.add_paths(); import shutil; print('ffmpeg at:', shutil.which('ffmpeg')); print('ffprobe at:', shutil.which('ffprobe'))"

COPY app.py /app/app.py
WORKDIR /app

RUN mkdir -p /tmp/parliament

ENV PORT=8000
EXPOSE ${PORT}

CMD uvicorn app:app --host 0.0.0.0 --port ${PORT}
