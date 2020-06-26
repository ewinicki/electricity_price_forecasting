import os
import json
import pandas as pd
from Results import TrainResults, TestResults
from write_excel import write_excel

class Model(dict):
    def __init__(self, name, category, classification, parameters):
        self["name"] = name
        self["category"] = category
        self["classification"] = classification
        self["parameters"] = parameters
        self["results"] = {}

    def __init__(self, model_json):
        with open(model_json, 'r') as model_file:
            model = json.load(model_file)

        self["name"] = model["name"]
        self["category"] = model["category"]
        self["classification"] = model["classification"]
        self["parameters"] = model["parameters"]
        self["results"] = {}

    def to_excel(self, fn):
        if os.path.isfile(fn):
            sheets = pd.read_excel(fn, sheet_name=None, index_col=0)
        else:
            sheets = {}

        data = pd.DataFrame(self.parameters.items())

        sheets["parameters"] = data

        writer = pd.ExcelWriter(fn, engine='xlsxwriter')
        workbook = writer.book
        formatting = workbook.add_format({
            'font_name':'Times New Roman'
            })
        formatting.set_border()

        for sheet, data in sheets.items():
            data.to_excel(writer, sheet_name=sheet)
            writer.sheets[sheet].set_column(0, data.shape[1], None, formatting)

        writer.save()
        writer.close()

    def __getattr__(self, key):
        if key in self:
            return self[key]
        else:
            raise AttributeError("{} does not exist".format(key))

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        if key in self:
            del self[key]
        else:
            raise AttributeError("{} does not exist".format(key))
