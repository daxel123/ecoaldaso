from flet import *
import json

def update_record_list(record_list, object_records, page):
    # Sort object_records by 'timestamp' in descending order
    sorted_records = sorted(object_records, key=lambda r: r['timestamp'], reverse=True)
    
    record_list.rows.clear()
    for record in sorted_records:
        label, confidence, points, timestamp = record['label'], record['confidence'], record['points'], record['timestamp']
        record_list.rows.append(
            DataRow(
                cells=[
                    DataCell(Text(label)),
                    DataCell(Text(f"{confidence:.2f}")),
                    DataCell(Text(f"{points}")),
                    DataCell(Text(timestamp)),
                ]
            )
        )
    page.update()


def load_records_from_json():
    with open('records.json', 'r') as file:
        return json.load(file)

def insert_record_into_json(record):
    with open('records.json', 'r+') as file:
        data = json.load(file)
        data.append(record)
        file.seek(0)
        json.dump(data, file, indent=4)
