import json
import csv
import requests

url = "https://huggingface.co/allenai/Olmo-3-1025-7B/raw/main/vocab.json"
response = requests.get(url)
vocab = response.json()

with open("vocab/vocab_olmo3.csv", "w", newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["token", "id"])
    for token, idx in vocab.items():
        writer.writerow([token, idx])