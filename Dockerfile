# v4 - fix static_ffmpeg binary permissions
FROM python:3.12-slim

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Force static_ffmpeg to extract its binaries, then make them executable
RUN python -c "import static_ffmpeg; static_ffmpeg.add_paths()" && \
    chmod +x /usr/local/lib/python3.12/site-packages/static_ffmpeg/bin/linux/ffmpeg && \
    chmod +x /usr/local/lib/python3.12/site-packages/static_ffmpeg/bin/linux/ffprobe && \
    /usr/local/lib/python3.12/site-packages/static_ffmpeg/bin/linux/ffmpeg -version && \
    /usr/local/lib/python3.12/site-packages/static_ffmpeg/bin/linux/ffprobe -version

COPY app.py /app/app.py
WORKDIR /app

RUN mkdir -p /tmp/parliament

ENV PORT=8000
EXPOSE ${PORT}

CMD uvicorn app:app --host 0.0.0.0 --port ${PORT}
