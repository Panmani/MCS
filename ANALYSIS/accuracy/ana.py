import parse
import matplotlib.pyplot as plt
import numpy as np

TRAIN_ACC_PATTERN = 'Rank[{}]Epoch[{}] Batch [{}]	Speed: {} samples/s ETA: {} d {} h {} m	Data: {} Tran: {} F: {} B: {} O: {} M: {}	Train-Acc={},	AnsLoss={},	CNNRegLoss={},{}'

# VAL_ACC_PATTERN = 'Best Val Acc: {}, Epoch: {}'
VAL_ACC_PATTERN = 'Epoch[{}] 	Val-Acc={},{}'

def get_train_acc_history(filename):
    train_acc_history = []
    with open(filename) as file:
        for line in file:
            parse_list = parse.parse(TRAIN_ACC_PATTERN, line)
            if parse_list is not None:
                rank, epoch, batch, _, _, _, _, _, _, _, _, _, _, train_acc, ans_loss, cnnregloss,_ = parse_list
                rank, epoch, batch, train_acc, ans_loss, cnnregloss = int(rank), int(epoch), int(batch), float(train_acc), float(ans_loss), float(cnnregloss)
                print(rank, epoch, batch, train_acc, ans_loss, cnnregloss)
                if rank == 0:
                    train_acc_history.append(train_acc)
    return train_acc_history

def get_val_acc_history(filename):
    val_acc_history = []
    have_seen = []
    with open(filename) as file:
        for line in file:
            parse_list = parse.parse(VAL_ACC_PATTERN, line)
            if parse_list is not None:
                # val_acc, epoch = parse.parse(VAL_ACC_PATTERN, line)
                # val_acc, epoch = float(val_acc), int(epoch)
                epoch, val_acc, _ = parse.parse(VAL_ACC_PATTERN, line)
                epoch, val_acc = int(epoch), float(val_acc)
                print(val_acc, epoch)
                if epoch not in have_seen:
                    have_seen.append(epoch)
                    val_acc_history.append(val_acc)
    return val_acc_history

vcr_comet_train_acc = get_train_acc_history("comet.txt")
vcr_train_acc = get_train_acc_history("vcr.txt")
plt.plot(vcr_comet_train_acc)
plt.plot(vcr_train_acc)
plt.ylabel('Train Accuracy')
plt.xlabel('Steps')
plt.legend(['With COMET', 'Only VCR'], loc='lower right')
plt.show()

vcr_comet_val_acc = get_val_acc_history('comet.txt')
vcr_val_acc = get_val_acc_history('vcr.txt')
plt.plot(vcr_comet_val_acc)
plt.plot(vcr_val_acc)

bottom = 0.6
leftmost = -0.5
gap = 0.003

comet_epoch = 7
comet_best_acc = 0.7450445294380188
plt.vlines(comet_epoch, bottom, comet_best_acc, linestyle="dotted", color='g')
plt.hlines(comet_best_acc, leftmost, comet_epoch, linestyle="dotted", color='g')
plt.text(comet_epoch + gap, comet_best_acc + gap, 'Best-Epoch:{}, Acc:{:04.2f}%'.format(comet_epoch, comet_best_acc * 100), fontsize=7.5, color='b')

vcr_epoch = 15
vcr_best_acc = 0.712806224822998
plt.vlines(vcr_epoch, bottom, vcr_best_acc, linestyle="dotted", color='g')
plt.hlines(vcr_best_acc, leftmost, vcr_epoch, linestyle="dotted", color='g')
plt.text(vcr_epoch + gap, vcr_best_acc + gap, 'Best-Epoch:{}, Acc:{:04.2f}%'.format(vcr_epoch, vcr_best_acc * 100), fontsize=7.5, color='b')

plt.ylabel('Validation Accuracy')
plt.xlabel('Epochs')
plt.xticks(range(len(vcr_comet_val_acc) + 1))
plt.ylim(0.6, 0.76)
plt.xlim(leftmost, 21)
plt.legend(['With COMET', 'Only VCR'], loc='lower right')
plt.show()

print(np.mean(vcr_comet_train_acc[-400:]))
print(np.mean(vcr_train_acc[-400:]))

print(vcr_comet_val_acc)
print(vcr_val_acc)
