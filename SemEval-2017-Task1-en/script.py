with open('./Output_Dev/STS.gs.dev.en-en.txt', 'r', encoding='utf-8') as f:
    L_dev = len(f.read().strip().split('\n'))

with open('./Output_Test/STS.gs.test.en-en.txt', 'r', encoding='utf-8') as f:
    L_test = len(f.read().strip().split('\n'))

dev = [str(0)]*L_dev
test = [str(0)]*L_test
with open('./Output_Dev/STS.random.dev.en-en.txt', 'w', encoding='utf-8') as f:
    f.write("\n".join(dev))
with open('./Output_Test/STS.random.test.en-en.txt', 'w', encoding='utf-8') as f:
    f.write("\n".join(test))
