import os
from constants import Base

MAX_RANGE = 1000


def get_next_file_range(directory, base_filename):
    max_range = -1

    # Iterate over files in the directory
    for file_name in os.listdir(directory):
        # Check if the file matches the base filename pattern
        if file_name.startswith(base_filename + "_"):
            # Extract the range number
            file_name = file_name.split(".")[0]
            try:
                file_range = int(file_name.split("_")[1])
                max_range = max(max_range, file_range)
            except ValueError:
                continue

    # Increment the max_range to get the next available range
    next_range = (max_range + 1) % MAX_RANGE

    return next_range


def get_xlsx_write_file(trainer="optuna"):
    """
    Return the next XLSX file name to be created in the directory
    :param trainer:
    :return:
    """
    base_file_name = "PredProbs"
    directory = os.path.join(Base.BASE_XLSX_PATH, trainer)
    next_range = get_next_file_range(directory, base_file_name)
    return os.path.join(directory, base_file_name + f"_{next_range}.xlsx")


def write_classifer_prediction_xlsx(workbook, model_scores, targets, predicted):
    """
    Writes the xlsx file using openpyxl
    :param workbook:
    :param model_scores:
    :param targets:
    :param predicted:
    :return:
    """
    headers = ["Prediction", "TargetVal", "PredVal", "Correctness"]

    for classification in range(0, 3):
        sheet_name = f'class_{classification}'
        prediction_scores = model_scores[:, classification]

        if sheet_name in workbook.sheetnames:
            worksheet = workbook[sheet_name]
        else:
            worksheet = workbook.create_sheet(sheet_name)
            for col, header in enumerate(headers):
                worksheet.cell(row=1, column=col + 1, value=header)

        for i, score in enumerate(prediction_scores):
            worksheet.append([score.item(), int(targets[i] == classification), int(predicted[i] == classification),
                              int(targets[i] == predicted[i])])


def write_linear_prediction_xlsx(workbook, model_scores, targets, predicted, sheet_name):
    """
    Writes the xlsx file using openpyxl
    :param workbook:
    :param model_scores:
    :param targets:
    :param predicted:
    :return:
    """
    headers = ["Prediction", "TargetVal", "PredVal"]

    if sheet_name in workbook.sheetnames:
        worksheet = workbook[sheet_name]
    else:
        worksheet = workbook.create_sheet(sheet_name)
        for col, header in enumerate(headers):
            worksheet.cell(row=1, column=col + 1, value=header)

    for i, score in enumerate(model_scores):
        worksheet.append([score.item(), int(targets[i]), predicted[i].item()])


def read_linear_prediction_xlsx(workbook, sheet_name, col_index=0, col_name=""):
    headers = {"Prediction": "A",
               "TargetVal": "B",
               "PredVal": "C"}

    if col_name != "":
        col_index = headers[col_name]

    if sheet_name not in workbook.sheetnames:
        return []

    worksheet = workbook[sheet_name]
    col_data = list(map(int, [cell.value for cell in worksheet[col_index]][1:]))
    return col_data