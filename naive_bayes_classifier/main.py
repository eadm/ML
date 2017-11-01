import ml
import reader

cv = ml.create_cv_from_blocks(reader.read_blocks("pu1"))


print ml.create_dict(cv[0]["test"], count_twice=False)
