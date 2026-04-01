import torch

ckpt_path = '/data/ssd2/data_rishabh/auto-avsr/authors/LRS3_V_WER19.1/model.pth'
ckpt = torch.load(ckpt_path, map_location='cpu')

if 'state_dict' in ckpt:
    state_dict = ckpt['state_dict']
else:
    state_dict = ckpt

# Print first 20 keys
print('First 20 keys:')
for i, k in enumerate(state_dict.keys()):
    if i < 20:
        print(k)
    else:
        break

# Save all keys to a file
with open('ckpt_keys.txt', 'w') as f:
    for k in state_dict.keys():
        f.write(k + '\n')

print('All keys saved to ckpt_keys.txt')
