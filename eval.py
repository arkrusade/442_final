from tqdm import tqdm
import os
import PIL.Image as Image
import torch
from torch.autograd import Variable
from data import data_transforms
from model import Network

outfile = "outfile.csv"
test_dir = '/test_images'
model_file = "model.pth"

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


state_dict = torch.load(model_file)
model = Network()
model.load_state_dict(state_dict)
model.eval()

output_file = open(outfile, "w")
output_file.write("Filename,ClassId\n")
for f in tqdm(os.listdir(test_dir)):
    if 'ppm' in f:
        data = data_transforms(pil_loader(test_dir + '/' + f))
        data = data.view(1, data.size(0), data.size(1), data.size(2))
        data = Variable(data, volatile=True)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]

        file_id = f[0:5]
        output_file.write("%s,%d\n" % (file_id, pred))

output_file.close()

print("Succesfully wrote " + outfile)
