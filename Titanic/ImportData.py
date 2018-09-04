import csv

def loadCSV(file, header = False):
    """Loads a CSV file and converts all floats and ints into basic datatypes."""

    def convertTypes(s):
        s = s.strip()
        try:
            return float(s) if '.' in s else int(s)
        except ValueError:
            return s

    reader = csv.reader(open(file, 'rt')) # Generator
    return [[convertTypes(item) for item in row] for row in reader]