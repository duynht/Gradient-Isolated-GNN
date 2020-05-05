import torch
import time
import numpy as np

#### own modules
from GreedyInfoMax.utils import logger
from GreedyInfoMax.vision.arg_parser import arg_parser
from GreedyInfoMax.vision.models import load_vision_model
from GreedyInfoMax.vision.data import get_dataloader


def validate(opt, model, test_loader):
    total_step = len(test_loader)

    loss_epoch = [0 for i in range(opt.model_splits)]
    starttime = time.time()

    for step, (img, label) in enumerate(test_loader):

        model_input = img.to(opt.device)
        label = label.to(opt.device)

        loss, _, _, _ = model(model_input, label, n=opt.train_module)
        loss = torch.mean(loss, 0)

        loss_epoch += loss.data.cpu().numpy()

    for i in range(opt.model_splits):
        print(
            "Validation Loss Model {}: Time (s): {:.1f} --- {:.4f}".format(
                i, time.time() - starttime, loss_epoch[i] / total_step
            )
        )

    validation_loss = [x/total_step for x in loss_epoch]
    return validation_loss


def train(opt, models):
    num_patches = opt.grid_dims * opt.grid_dims

    total_step = len(train_loader)
    for i in range(num_patches):
        models[i].module.switch_calc_loss(True)

    print_idx = 100

    starttime = time.time()
    cur_train_module = opt.train_module

    for epoch in range(opt.start_epoch, opt.num_epochs + opt.start_epoch):

        loss_epoch = [0 for i in range(opt.model_splits)]
        loss_updates = [1 for i in range(opt.model_splits)]

        for step, (full_img, label) in enumerate(train_loader):

            # split normalized images
            batch_size, num_channels, img_h, img_w = full_img.shape
            components = []
            step_h = img_h // opt.grid_dims
            step_w = img_w // opt.grid_dims
            for height in range(0, img_h, step_h):
                for width in range(0, img_w, step_w):
                    components.append(full_img[:, :, height:height+step_h, width:width+step_w])  

            for component_index in range(num_patches):
                img = components[component_index]
                if step % print_idx == 0:
                    print(
                        "Epoch [{}/{}], Step [{}/{}], Training Block: {}, Time (s): {:.1f}".format(
                            epoch + 1,
                            opt.num_epochs + opt.start_epoch,
                            step,
                            total_step,
                            cur_train_module,
                            time.time() - starttime,
                        )
                    )

                starttime = time.time()

                model_input = img.to(opt.device)
                label = label.to(opt.device)

                loss, _, _, accuracy = models[component_index](model_input, label, n=cur_train_module)
                loss = torch.mean(loss, 0) # take mean over outputs of different GPUs
                accuracy = torch.mean(accuracy, 0)

                if cur_train_module != opt.model_splits and opt.model_splits > 1:
                    loss = loss[cur_train_module].unsqueeze(0)

                # loop through the losses of the modules and do gradient descent
                for idx, cur_losses in enumerate(loss):
                    if len(loss) == 1 and opt.model_splits != 1:
                        idx = cur_train_module

                    models[component_index].zero_grad()

                    if idx == len(loss) - 1:
                        cur_losses.backward()
                    else:
                        cur_losses.backward(retain_graph=True)
                    optimizers[component_index][idx].step()

                    print_loss = cur_losses.item()
                    print_acc = accuracy[idx].item()
                    if step % print_idx == 0:
                        print("\t \t Loss: \t \t {:.4f}".format(print_loss))
                        if opt.loss == 1:
                            print("\t \t Accuracy: \t \t {:.4f}".format(print_acc))

                    loss_epoch[idx] += print_loss
                    loss_updates[idx] += 1

        for component_index in range(num_patches):
            if opt.validate:
                validation_loss = validate(opt, models[component_index], test_loader) #test_loader corresponds to validation set here
                logs.append_val_loss(validation_loss)

            logs.append_train_loss([x / loss_updates[idx] for idx, x in enumerate(loss_epoch)])
            logs.create_log(models[component_index], epoch=epoch, optimizer=optimizers[component_index], component_idx=component_index)


if __name__ == "__main__":

    opt = arg_parser.parse_args()
    arg_parser.create_log_path(opt)
    opt.training_dataset = "unlabeled"

    # random seeds
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    if opt.device.type != "cpu":
        torch.backends.cudnn.benchmark = True

    num_patches = opt.grid_dims * opt.grid_dims

    # load model
    models, optimizers = [None for i in range(num_patches)], [None for i in range(num_patches)]
    for i in range(num_patches):
        models[i], optimizers[i] = load_vision_model.load_model_and_optimizer(opt)

    logs = logger.Logger(opt)

    train_loader, _, supervised_loader, _, test_loader, _ = get_dataloader.get_dataloader(
        opt
    )

    if opt.loss == 1:
        train_loader = supervised_loader

    try:
        # Train the model
        train(opt, models)

    except KeyboardInterrupt:
        print("Training got interrupted, saving log-files now.")

    logs.create_log(models)

