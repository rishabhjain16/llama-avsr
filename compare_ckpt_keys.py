import torch

ckpt1 = torch.load('/data/ssd2/data_rishabh/auto-avsr/authors/LRS3_V_WER19.1/model.pth', map_location='cpu')
ckpt2 = torch.load('/data/ssd2/data_rishabh/auto-avsr/authors/vsr_trlrs3_23h_base.pth', map_location='cpu')

def get_keys(ckpt):
    if 'state_dict' in ckpt:
        return set(ckpt['state_dict'].keys())
    return set(ckpt.keys())

keys1 = get_keys(ckpt1)
keys2 = get_keys(ckpt2)

print("Keys unique to LRS3_V_WER19.1/model.pth:")
for k in sorted(keys1 - keys2):
    print(k)

print("\nKeys unique to vsr_trlrs3_23h_base.pth:")
for k in sorted(keys2 - keys1):
    print(k)