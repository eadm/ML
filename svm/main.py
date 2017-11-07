import reader
import ml

FOLDS = 10

points, classes = reader.read_data("chips.txt")

folds = ml.folds(points, classes, FOLDS, shuffle=True)
