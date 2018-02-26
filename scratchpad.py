import blib

X1_train, X2_train, y_train = blib.text.from_csv('quora_duplicate_questions_dev.csv', cols=[0,1,2], skip_header=True)

print(len(X1_train))
print(len(X2_train))
print(len(y_train))

print(X1_train[0])
print(X2_train[0])
print(y_train[0])