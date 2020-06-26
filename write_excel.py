import os
import pandas as pd

def write_excel(data, fn, sheet_name):
    if os.path.isfile(fn):
        sheets = pd.read_excel(fn, sheet_name=None, index_col=0)
    else:
        sheets = {}

    sheets[sheet_name] = data

    writer = pd.ExcelWriter(fn, engine='xlsxwriter')
    workbook = writer.book
    formatting = workbook.add_format({
        'num_format':'#.000',
        'font_name':'Times New Roman'
        })
    formatting.set_border()

    for sheet, data in sheets.items():
        data.to_excel(writer, sheet_name=sheet)
        writer.sheets[sheet].set_column(0, data.shape[1], None, formatting)

    writer.save()
    writer.close()

