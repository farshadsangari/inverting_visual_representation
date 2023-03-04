import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn

######  Local packages  ######
import learning as learning
import models as models
import dataloader
import util as util


def main(args):

    train_pathes, val_pathes, _ = dataloader.train_val_pathes(
        args.all_data_pathes, args.regex_for_category
    )
    data_transforms = dataloader.data_transform()

    train_dataloader, val_dataloader = dataloader.train_val_loader(
        train_pathes, val_pathes, data_transforms, args.batch_size
    )

    model = models.NetConv2()

    criterion = nn.MSELoss()

    optimizer = optim.Adam(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=args.lr,
        betas=args.betas,
        weight_decay=args.weight_decay,
    )

    # Loading Model
    if args.ckpt_load_path is not None:
        print("******  Loading Model   ******")
        model, optimizer = util.load_model(
            ckpt_path=args.ckpt_load_path, model=model, optimizer=optimizer
        )

    model = model.to(args.device)
    criterion = criterion.to(args.device)

    # Schedular
    lr_scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # Train the model(Train and Validation Steps)
    model, optimizer = learning.Train_mode(
        model=model,
        device=args.device,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        num_epochs=args.num_epochs,
        saving_checkpoint_path=args.ckpt_save_path,
        saving_prefix=args.ckpt_prefix,
        saving_checkpoint_freq=args.ckpt_save_freq,
        report_path=args.report_path,
    )

    return model


if __name__ == "__main__":
    args = util.get_args()
    main(args)
