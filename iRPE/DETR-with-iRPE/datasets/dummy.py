import os
paths = [os.path.join(os.getcwd(), f) for f in os.listdir('.') if f.endswith('.py')]
for path in paths:
    print(path)