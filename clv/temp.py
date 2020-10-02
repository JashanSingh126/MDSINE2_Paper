import pickle

with open("data/bucci/T_cdiff.pkl", "rb") as output_file:
    d = pickle.load(output_file)

print(d)
print(type(d))


