#%% Loss and accuracy plotting
import pandas as pd
import matplotlib.pyplot as plt

reader= pd.read_csv('D:/CSAI_thesis/denoised_aug_lr=0.002_attemp/log_aug__den_lr=0.002_acc_train.csv', chunksize=620)
final= pd.DataFrame()
for file in reader:
    file2 =file.mean()
    final = final.append(file2, ignore_index= True)
    
#%%
import matplotlib.pyplot as plt

print(final.shape)

plt.plot(range(0,50),final['d_loss1'])
plt.plot(range(0,50),final['d_loss2'], alpha=0.7)
plt.plot(range(0,50),final['g_loss'])
plt.title('train_loss')
plt.xlabel('Epochs')
plt.ylabel('loss_value')
plt.legend(['d_loss1','d_loss2','g_loss'])

plt.show()
#%%
import matplotlib.pyplot as plt

print(final.shape)

plt.plot(range(0,50),final['acc_fake'])
#plt.plot(range(0,50),final['d_loss2'], alpha=0.7)
plt.plot(range(0,50),final['acc_real'])
plt.title('discriminator_train_acc')
plt.xlabel('Epochs')
plt.ylabel('acc_value')
plt.legend(['acc_fake','acc_real'])

plt.show()