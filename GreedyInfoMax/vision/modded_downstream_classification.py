import torch
import numpy as np
import time
import os
import copy

# own modules
from GreedyInfoMax.vision.data import get_dataloader
from GreedyInfoMax.vision.arg_parser import arg_parser
from GreedyInfoMax.vision.models import load_vision_model, SmallModel
from GreedyInfoMax.utils import logger, utils


def train_logistic_regression(opt, context_models, classification_model, train_loader):
    total_step = len(train_loader)
    classification_model.train()

    starttime = time.time()

    for epoch in range(opt.num_epochs):
        epoch_acc1 = 0
        epoch_acc3 = 0

        loss_epoch = 0
        for step, batch in enumerate(train_loader):
            full_img, desc, target = batch['img'], batch['desc'], batch['label']
            classification_model.zero_grad()

            batch_size, num_channels, img_h, img_w = full_img.shape
            patches = []
            # first patch
            patches.append(full_img[:, :, :img_h//2, :img_w//2])
            # second patch
            patches.append(full_img[:, :, :img_h//2, img_w//2:])
            # third patch
            patches.append(full_img[:, :, img_h//2:, :img_w//2])
            # fourth patch
            patches.append(full_img[:, :, img_h//2:, img_w//2:])

            model_inputs = []
            for patch_idx in range(4):
                model_inputs.append(patches[patch_idx].to(opt.device))

            with torch.no_grad():
                z = None
                for patch_idx in range(4):
                    if not opt.use_simple_resnet:
                        _, _, patch_z, _ = context_model(
                            model_inputs[patch_idx], target)
                    else:
                        patch_z = context_model(model_inputs[patch_idx])
                    if z is None:
                        z = copy.deepcopy(patch_z)
                        z.to(opt.device)
                    else:
                        z += patch_z
                z /= 4
            z = z.detach()  # double security that no gradients go to representation learning part of model
            # print(z.shape)
            # z = torch.mean(torch.mean(z, -1, True), -2, True)
            # print(z.shape)
            fusion_vector = {
                'img': z,
                'desc': desc.to(opt.device)
            }

            prediction = classification_model(fusion_vector)

            target = target.to(opt.device)
            loss = criterion(prediction, target)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate accuracy
            acc1, acc3 = utils.accuracy(prediction.data, target, topk=(1, 3))
            epoch_acc1 += acc1
            epoch_acc3 += acc3

            sample_loss = loss.item()
            loss_epoch += sample_loss

            if step % 10 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Time (s): {:.1f}, Acc1: {:.4f}, Acc3: {:.4f}, Loss: {:.4f}".format(
                        epoch + 1,
                        opt.num_epochs,
                        step,
                        total_step,
                        time.time() - starttime,
                        acc1,
                        acc3,
                        sample_loss,
                    )
                )
                starttime = time.time()

        if opt.validate:
            # validate the model - in this case, test_loader loads validation data
            val_acc1, _, val_loss = test_logistic_regression(
                opt, context_model, classification_model, test_loader
            )
            logs.append_val_loss([val_loss])

        print("Overall accuracy for this epoch: ", epoch_acc1 / total_step)
        logs.append_train_loss([loss_epoch / total_step])
        logs.create_log(
            context_model,
            epoch=epoch,
            classification_model=classification_model,
            accuracy=epoch_acc1 / total_step,
            acc5=epoch_acc3 / total_step,
        )


def test_logistic_regression(opt, context_model, classification_model, test_loader):
    total_step = len(test_loader)
    context_model.eval()
    classification_model.eval()

    starttime = time.time()

    loss_epoch = 0
    epoch_acc1 = 0
    epoch_acc3 = 0

    for step, batch in enumerate(test_loader):
        full_img, desc, target = batch['img'], batch['desc'], batch['label']
        classification_model.zero_grad()
        batch_size, num_channels, img_h, img_w = full_img.shape
        patches = []
        # first patch
        patches.append(full_img[:, :, :img_h//2, :img_w//2])
        # second patch
        patches.append(full_img[:, :, :img_h//2, img_w//2:])
        # third patch
        patches.append(full_img[:, :, img_h//2:, :img_w//2])
        # fourth patch
        patches.append(full_img[:, :, img_h//2:, img_w//2:])
        model_inputs = []
        for patch_idx in range(4):
            model_inputs.append(patches[patch_idx].to(opt.device))

        with torch.no_grad():
            z = None
            for patch_idx in range(4):
                if not opt.use_simple_resnet:
                    _, _, patch_z, _ = context_model(
                        model_inputs[patch_idx], target)
                else:
                    patch_z = context_model(model_inputs[patch_idx])
                if z is None:
                    z = copy.deepcopy(patch_z)
                    z.to(opt.device)
                else:
                    z += patch_z
            z /= 4

        z = z.detach()  # double security that no gradients go to representation learning part of model
        # z = torch.mean(torch.mean(z, -1, True), -1, True)
        fusion_vector = {
                'img': z,
                'desc': desc.to(opt.device)
            }

        prediction = classification_model(fusion_vector)

        target = target.to(opt.device)
        loss = criterion(prediction, target)

        # calculate accuracy
        acc1, acc3 = utils.accuracy(prediction.data, target, topk=(1, 3))
        epoch_acc1 += acc1
        epoch_acc3 += acc3

        sample_loss = loss.item()
        loss_epoch += sample_loss

        if step % 10 == 0:
            print(
                "Step [{}/{}], Time (s): {:.1f}, Acc1: {:.4f}, Acc3: {:.4f}, Loss: {:.4f}".format(
                    step, total_step, time.time() - starttime, acc1, acc3, sample_loss
                )
            )
            starttime = time.time()

    print("Testing Accuracy: ", epoch_acc1 / total_step)
    return epoch_acc1 / total_step, epoch_acc3 / total_step, loss_epoch / total_step


if __name__ == "__main__":

    opt = arg_parser.parse_args()

    add_path_var = "linear_model"

    arg_parser.create_log_path(opt, add_path_var=add_path_var)
    opt.training_dataset = "train"

    # random seeds
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    # load pretrained model
    context_models = []

    if not opt.use_simple_resnet:
        for patch_idx in range(4):
            context_model, _ = load_vision_model.load_model_and_optimizer(
                opt, reload_model=True, calc_loss=False, patch_idx=patch_idx
            )
            context_model.module.switch_calc_loss(False)
            context_models.append(context_model)
            
        # model_type=2 is supervised model which trains entire architecture; otherwise just extract features
        if opt.model_type != 2:
            for i in range(4):
                context_models[i].eval()
    else:
        context_models.append(SmallModel.ResNetModel(opt))

    _, _, train_loader, _, test_loader, _ = get_dataloader.get_dataloader(opt)

    classification_model = load_vision_model.load_fusion_classification_model(opt)

    if opt.model_type == 2:
        params = list(context_model.parameters()) + \
            list(classification_model.parameters())
    else:
        params = classification_model.parameters()

    optimizer = torch.optim.Adam(params)
    criterion = torch.nn.CrossEntropyLoss()

    logs = logger.Logger(opt)

    try:
        # Train the model
        train_logistic_regression(
            opt, context_models, classification_model, train_loader)

        # Test the model
        acc1, acc5, _ = test_logistic_regression(
            opt, context_model, classification_model, test_loader
        )

    except KeyboardInterrupt:
        print("Training got interrupted")

    logs.create_log(
        context_model,
        classification_model=classification_model,
        accuracy=acc1,
        acc5=acc5,
        final_test=True,
    )
    torch.save(
        context_model.state_dict(), os.path.join(opt.log_path, "context_model.ckpt")
    )
