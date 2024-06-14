FROM python:3.11.9-bookworm
WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY . .
EXPOSE 3000
CMD ["python", "-u", "src/main.py"]