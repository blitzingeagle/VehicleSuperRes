import numpy as np
import cv2
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda")

model = nn.Sequential(
    nn.Conv2d(3, 16, (3, 3), padding=(1, 1)),
    nn.LeakyReLU(0.1),
    nn.Conv2d(16, 32, (3, 3), padding=(1, 1)),
    nn.LeakyReLU(0.1),
    nn.Conv2d(32, 64, (3, 3), padding=(1, 1)),
    nn.LeakyReLU(0.1),
    nn.Conv2d(64, 128, (3, 3), padding=(1, 1)),
    nn.LeakyReLU(0.1),
    nn.Conv2d(128, 128, (3, 3), padding=(1, 1)),
    nn.LeakyReLU(0.1),
    nn.Conv2d(128, 256, (3, 3), padding=(1, 1)),
    nn.LeakyReLU(0.1),
    nn.ConvTranspose2d(256, 3, (4, 4), (2, 2), (1, 1), (0, 0), bias=False),
    nn.Sigmoid()
).to(device)

learning_rate = 0.01
learning_rate_decay = 0.85
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
loss_fn = F.mse_loss
batches = 50
epochs = 250

log_interval = 1

def train(epoch, batch, train_data):
    model.train()
    for batch_idx, (input, target) in enumerate(train_data):
        input, target = input.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print("Train Epoch: {} Batch: {}:{} Rate:{} \tLoss: {:.6f}".format(epoch, batch, batch_idx, learning_rate, loss.item()))



if __name__ == "__main__":
    for epoch in range(1, epochs+1):
        img_no = 1
        for batch in range(batches):
            batch_dir = "train/batches/batch_{:02d}".format(batch)
            train_data = []

            while True:
                path_in = "{}/input_{:04d}.bmp".format(batch_dir, img_no)
                path_out = "{}/output_{:04d}.png".format(batch_dir, img_no)

                img_in = cv2.imread(path_in)
                img_out = cv2.imread(path_out)

                if img_in is None or img_out is None:
                    break

                img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
                img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)

                img_in = np.swapaxes(np.swapaxes(np.array(img_in, dtype=float), 0, 2), 1, 2) / 255.0
                img_out = np.swapaxes(np.swapaxes(np.array(img_out, dtype=float), 0, 2), 1, 2) / 255.0

                shape_in = (1,) + img_in.shape
                shape_out = (1,) + img_out.shape

                input = torch.from_numpy(img_in.reshape(shape_in)).float().cuda()
                output = torch.from_numpy(img_out.reshape(shape_out)).float().cuda()

                train_data.append((input, output))
                img_no += 1

            train(epoch, batch, train_data)

        learning_rate *= learning_rate_decay
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

        print(model.state_dict())
        if not os.path.exists("trained-weights"): os.makedirs("trained-weights")
        torch.save(model.state_dict(), "trained-weights/weights{:03d}.pth".format(epoch))

    torch.save(model.state_dict(), "weights.pth".format(epoch))
