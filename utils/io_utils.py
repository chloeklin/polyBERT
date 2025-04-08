import ast
import os
import csv

def load_txt_to_list(file_path):
    combined = []
    if os.path.isfile(file_path):
        with open(file_path, "r") as file:
            for line in file:
                combined.extend(ast.literal_eval(line.strip()))
    else:
        print("File does not exist.")
    return combined


def write_row_to_csv(file_path, row):
    updated = False
    rows = []

    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            reader = csv.reader(file)
            header = next(reader, None)
            rows.append(header)
            for existing_row in reader:
                if existing_row[0] == row[0]:
                    rows.append(row)
                    updated = True
                else:
                    rows.append(existing_row)

    if not updated:
        if not os.path.exists(file_path):
            rows.append(['Pretrain size', 'f1-score'])
        rows.append(row)

    with open(file_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(rows)
