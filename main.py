import torch
from torch.utils import data
import os
import numpy as np
from utils import trainer, visualizer, cagan_dataset, options, IS_score
from tqdm import tqdm

opt = options.GatherOptions().parse()
cagan_dataset = cagan_dataset.CAGAN_Dataset(opt)
if (opt.lr_policy == 'lambda') and (opt.niter_decay == -1):
    opt.niter_decay = opt.epoch*len(cagan_dataset) - opt.niter
model = trainer.CAGAN_Trainer(opt)

if opt.mode == "train":

    train_dataloader = data.DataLoader(cagan_dataset, opt.batchsize, shuffle=True,
                                    num_workers=opt.num_workers, pin_memory=True)
    # calculate num of steps for decaying learning to zero at the end of training


    vis_custom = visualizer.Visualizer(opt.mode)
    vis_custom.reset_env()
    loss_save_path = os.path.join(opt.save_dir, 'loss.dict')
    step = 0
    if opt.resume:
        step = opt.step
        model.load_networks(step=step, load_netD=True)
        try:
            vis_custom.recover_loss(loss_save_path)
        except:
            print("Loss dict can not be found in %s"%opt.save_dir)

    for epoch in range(opt.epoch):
        for real in train_dataloader:
            step += 1
            model.set_input(real)
            model.optimize_parameters()

            loss_dict =model.get_current_losses()
            loss_str = '[step %d/epoch %d]'%(step, epoch+1)
            for key, data in loss_dict.items():
                if "sum" in key:
                    loss_str += key + ': ' + '%.3f'%(data) + ' '
            print(loss_str[:-2])
            vis_custom.plot_current_losses(step, loss_dict)
            vis_custom.plot_current_images(model.get_current_visuals(), 'real-fake-rec-alpha',len(real))
            if step % opt.save_image_freq == 0:
                model.save_current_visuals(step, opt.batchsize)
            if step % opt.save_model_freq == 0:
                model.save_networks(step)
                torch.save(vis_custom.plot_data, loss_save_path)
            if opt.lr_policy in ['lambda']:
                model.update_learning_rate()
        if opt.lr_policy not in ['lambda']:
            model.update_learning_rate()

    model.save_current_visuals(step, opt.batchsize)
    model.save_networks(step)
else:
    inception_model = IS_score.INCEPTION_V3()
    inception_model.eval()
    inception_model.to('cuda')
    test_dataloader = data.DataLoader(cagan_dataset, opt.batchsize, shuffle=False,
                                    num_workers=opt.num_workers, pin_memory=True)
    step = None
    if opt.step != 0:
        step = opt.step
    model.load_networks(step=step, load_dir=opt.model_dir)
    vis_custom = visualizer.Visualizer('test')
    vis_custom.reset_env()
    predictions = []
    step = 0
    for real in tqdm(test_dataloader):
        step += 1
        model.set_input(real)
        with torch.autograd.no_grad():
            model.netG_forward()
            pred = inception_model(model.output_dict['fake_outputs'][-1])
        predictions.append(pred.data.cpu().numpy())
        vis_custom.plot_current_images(model.get_current_visuals(),
                                    'real-fake-rec-alpha', len(real))
    predictions = np.concatenate(predictions, 0)
    mean, std = IS_score.compute_inception_score(predictions, 10)
    print('IS score --- mean: %.4f, std: %.4f'%(mean, std))
