import json
with open('platform/data/splits/train.json', encoding='utf-8') as f:
    data = json.load(f)
print(len(data))
print(type(data[0]['label']))
print(data[0]['label'])