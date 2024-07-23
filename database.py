import json

def load_records_from_json():
    with open('records.json', 'r',encoding="utf-8") as file:
        return json.load(file)

def insert_record_into_json(record):
    with open('records.json', 'r+', encoding="utf-8") as file:
        data = json.load(file)
        data.append(record)
        file.seek(0)
        json.dump(data, file, indent=4)
