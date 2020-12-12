out = ''
line_idx = 0
with open('pred_train_out.txt') as file:
    for line in file:
        if line_idx % 10000000 == 0:
            print(line_idx)
        if 'TRUNCATING' not in line and '150' not in line and line != '\n' and line != ' \n':
            out += line
            # print(repr(line), end = '')
            # print(line, end = '')
        line_idx += 1

print(out)

with open('test.txt', 'w') as outfile:
    outfile.write(out)
