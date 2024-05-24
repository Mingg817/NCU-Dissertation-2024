#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   trainer.py
@Time    :   2024/05/20
@Author  :   LI YIMING 
@Version :   1.0
@Site    :   https://github.com/Mingg817
@Desc    :   训练模型辅助函数
'''


import torch
from torch.utils.data import DataLoader
from d2l import torch as d2l


def GRU_FC_evaluate_loss(net, data_iter, loss_fn, forcast_length, **args):
    """Evaluate the loss of a model on the given dataset."""
    metric = d2l.Accumulator(2)  # Sum of losses, no. of examples
    for batch in data_iter:
        batch = {k: v.to("cuda") for k, v in batch.items() if isinstance(v, torch.Tensor)}
        out, _ = net(batch["x"], net.init_hidden(batch["x"].shape[0]), forcast_length, **args)
        l = loss_fn(out, batch["y_avg"])
        metric.add(l.detach().item(), 1)
    return metric[0] / metric[1]


def GRU_FC_trainer(
    model,
    num_steps,
    dataset,
    optimizer,
    loss_fn,
    batch_size,
    forcast_length=1,
    train_loss_sample_rate=1,
    test_loss_sample_rate=50,
    **args,
):
    train_loader = DataLoader(
        dataset["train"], batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True
    )
    test_loader = DataLoader(
        dataset["test"], batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True
    )
    animator = d2l.Animator(
        xlabel="steps",
        ylabel="loss",
        legend=["train loss", "test loss"],
        xlim=[test_loss_sample_rate // 2, num_steps],
        # ylim=[0, 1],
    )

    metric = d2l.Accumulator(2)
    steps = 0
    while steps < num_steps:
        for batch in train_loader:
            batch = {k: v.to("cuda") for k, v in batch.items() if isinstance(v, torch.Tensor)}

            # 训练
            model.train()
            # 将梯度清空
            optimizer.zero_grad()
            # 将数据放进去训练
            output, _ = model(
                batch["x"], model.init_hidden(batch["x"].shape[0]), forcast_length, **args
            )
            # 计算每次的损失函数
            # print(output.device, batch["y_avg"].device)
            train_loss = loss_fn(output, batch["y_avg"])
            # 反向传播
            train_loss.backward()
            # 优化器进行优化(梯度下降,降低误差)
            optimizer.step()

            steps += 1
            with torch.no_grad():

                if steps % train_loss_sample_rate == 0:
                    metric.add(train_loss, 1)

                if steps % test_loss_sample_rate == 0:
                    model.eval()
                    test_loss = GRU_FC_evaluate_loss(model, test_loader, loss_fn, forcast_length, **args)
                    animator.add(steps + 1, (metric[0] / metric[1], test_loss))
                    # animator.add(steps + 1, (train_loss.cpu().detach(), test_loss))
                    # metric.reset()

                if steps == num_steps:
                    break
    print(f"train loss {metric[0] / metric[1]:.6f}, test loss {test_loss:.6f}")


def NLDG_evaluate_loss(net, data_iter, loss_fn, forcast_length=-1, **args):
    """Evaluate the loss of a model on the given dataset."""
    metric = d2l.Accumulator(2)  # Sum of losses, no. of examples
    for batch in data_iter:
        batch = {k: v.to("cuda") for k, v in batch.items() if isinstance(v, torch.Tensor)}
        if forcast_length == -1:
            forcast_length = batch["y_hat"].shape[1]
        output = torch.zeros(batch["y_hat"].shape[0], forcast_length).to("cuda")
        h = net.init_hidden(batch["x"].shape[0])
        x = batch["x"]
        # batch["x"] -> [batch_size, seq_len]
        x_news = batch["x_news"]
        # batch["news"] -> [batch_size, seq_len, x_news_dim]
        for i in range(forcast_length):
            o, h = net(x, x_news, h)
            output[:, i] = o
            x = o.unsqueeze(dim=-1)
            x_news = batch["y_news"][:, i, :].unsqueeze(dim=1)
        out = torch.mean(output, dim=1)
        l = loss_fn(out, batch["y_avg"])
        metric.add(l.detach().item(), 1)
    return metric[0] / metric[1]


def NLDG_trainer(
    model,
    num_steps,
    dataset,
    optimizer,
    loss_fn,
    batch_size,
    forcast_length=-1,
    train_loss_sample_rate=1,
    test_loss_sample_rate=50,
    **args,
):

    train_loader = DataLoader(
        dataset["train"], batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True
    )
    test_loader = DataLoader(
        dataset["test"], batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True
    )
    animator = d2l.Animator(
        xlabel="steps",
        ylabel="loss",
        legend=["train loss", "test loss"],
        xlim=[test_loss_sample_rate // 2, num_steps],
        # ylim=[0, 1],
    )

    metric = d2l.Accumulator(2)
    metric_y = d2l.Accumulator(2)
    steps = 0

    if forcast_length == -1:
        forcast_length = dataset["y_hat"].shape[1]

    while steps < num_steps:
        for batch in train_loader:
            batch = {k: v.to("cuda") for k, v in batch.items() if isinstance(v, torch.Tensor)}

            # 训练
            model.train()
            # 将梯度清空
            optimizer.zero_grad()
            # 将数据放进去训练

            output = torch.zeros(batch["y_hat"].shape[0], forcast_length).to("cuda")
            # output -> [batch_size, forcast_length]
            h = model.init_hidden(batch["x"].shape[0])

            x = batch["x"]
            # batch["x"] -> [batch_size, seq_len]
            x_news = batch["x_news"]
            # batch["news"] -> [batch_size, seq_len, x_news_dim]

            # teacher forcing
            for i in range(forcast_length):
                o, h = model(x, x_news, h, **args)
                # o -> [batch_size]
                output[:, i] = o
                x = batch["y_hat"][:, i].unsqueeze(dim=-1)
                # x -> [batch_size, 1]
                x_news = batch["y_news"][:, i, :].unsqueeze(dim=1)

            output = torch.mean(output, dim=1)

            # 计算每次的损失函数
            train_loss = loss_fn(output, batch["y_avg"])
            # 反向传播
            train_loss.backward()
            # 优化器进行优化(梯度下降,降低误差)
            optimizer.step()

            steps += 1
            with torch.no_grad():

                if steps % train_loss_sample_rate == 0:
                    metric.add(train_loss, 1)

                if steps % test_loss_sample_rate == 0:
                    model.eval()
                    test_loss = NLDG_evaluate_loss(model, test_loader, loss_fn, forcast_length, **args)
                    animator.add(steps + 1, (metric[0] / metric[1], test_loss))
                    metric_y.add(test_loss, 1)
                    # animator.add(steps + 1, (train_loss.cpu().detach(), test_loss))
                    # metric.reset()

                if steps == num_steps:
                    break
    print(f"train loss {metric[0] / metric[1]:.6f}, test loss {test_loss:.6f}, avg test loss {metric_y[0] / metric_y[1]:.6f}")