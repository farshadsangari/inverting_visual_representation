import torch
import pandas as pd
from tqdm import tqdm
import os
import models as models
import util as util


def Train_mode(
    model,
    device,
    train_dataloader,
    val_dataloader,
    criterion,
    optimizer,
    lr_scheduler,
    num_epochs,
    saving_checkpoint_path,
    saving_prefix,
    saving_checkpoint_freq,
    report_path,
):

    report = pd.DataFrame(
        columns=[
            "mode",
            "epoch",
            "batch_index",
            "learning_rate",
            "loss_batch",
            "avg_epoch_loss_till_current_batch",
        ]
    )

    ###################################    Training mode     ##########################################
    for epoch in range(1, num_epochs + 1):
        avg_train_loss = util.AverageMeter()
        avg_val_loss = util.AverageMeter()
        mode = "train"
        model.train()
        # Loop for train batches
        loop_train = tqdm(
            enumerate(train_dataloader, 1),
            total=len(train_dataloader),
            desc="train",
            position=0,
            leave=True,
        )
        for batch_index, (inputs, labels) in loop_train:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            avg_train_loss.update(loss.item(), inputs.size(0))

            new_row = pd.DataFrame(
                {
                    "mode": mode,
                    "epoch": epoch,
                    "batch_index": batch_index,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "loss_batch": loss.item(),
                    "avg_epoch_loss_till_current_batch": avg_train_loss.avg,
                },
                index=[0],
            )
            report.loc[len(report)] = new_row.values[0]

            optimizer.step()

            loop_train.set_description(f"Train mode - epoch : {epoch}")
            loop_train.set_postfix(
                Loss_Train="{:.4f}".format(avg_train_loss.avg), refresh=True
            )
            if (epoch % saving_checkpoint_freq) == 0:
                util.save_model(
                    file_path=saving_checkpoint_path,
                    file_name=f"{saving_prefix}{epoch}.ckpt",
                    model=model,
                    optimizer=optimizer,
                )

        ################################    Validation mode   ##############################################
        model.eval()
        mode = "validation"
        with torch.no_grad():

            # Loop for val batches
            loop_val = tqdm(
                enumerate(val_dataloader, 1),
                total=len(val_dataloader),
                desc="val",
                position=0,
                leave=True,
            )
            for batch_index, (inputs, labels) in loop_val:

                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                avg_val_loss.update(loss.item(), inputs.size(0))

                new_row = pd.DataFrame(
                    {
                        "mode": mode,
                        "epoch": epoch,
                        "batch_index": batch_index,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "loss_batch": loss.item(),
                        "avg_epoch_loss_till_current_batch": avg_val_loss.avg,
                    },
                    index=[0],
                )
                report.loc[len(report)] = new_row.values[0]

                optimizer.zero_grad()
                loop_val.set_description(f"Validation mode - epoch : {epoch}")
                loop_val.set_postfix(
                    Loss_val="{:.4f}".format(avg_val_loss.avg), refresh=True
                )
        lr_scheduler.step()

    report.to_csv(os.path.join(report_path, f"report_training.csv"))
    return model, optimizer


def Inference_mode(
    model,
    device,
    test_dataloader,
    criterion,
    optimizer,
):

    model.eval()
    with torch.no_grad():
        avg_test_loss = util.AverageMeter()
        # Loop for test batches
        loop_test = tqdm(
            enumerate(test_dataloader, 1), total=len(test_dataloader), desc="val"
        )
        for batch_index, (inputs, labels) in loop_test:

            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            avg_test_loss.update(loss.item(), inputs.size(0))

            loop_test.set_description(f"Test mode")
            loop_test.set_postfix(
                Loss_test="{:.4f}".format(avg_test_loss.avg), refresh=True
            )
        print(f"@the end of training, Test loss value is : {avg_test_loss.avg}")
