import blib

X1_train, X2_train, y_train, X1_val, X2_val, y_val, X1_test, X2_test, y_test = blib.text.from_csv('../quora_duplicate_questions_dev.csv', cols=[0,1,2], skip_header=True, splits=[0.8,0.1, 0.1])


print("fin")
print(len(X1_train))
print(len(X2_train))
print(len(y_train))

print(len(X1_val))
print(len(X2_val))
print(len(y_val))

print(len(X1_test))
print(len(X2_test))
print(len(y_test))

print(X1_train[0])
print(X2_train[0])
print(y_train[0])

path = '../aclImdb/test'
folders = ['neg', 'pos']

X_train, y_train, X_val, y_val, X_test, y_test = blib.text.from_folders(path, folders, shuffle=True, splits=[0.8,0.1,0.1])

print(len(X_train))
print(len(y_train))
print(len(X_val))
print(len(y_val))
print(len(X_test))
print(len(y_test))

print(X_train[0])
print(y_train[0])

print(X_val[0])
print(y_val[0])