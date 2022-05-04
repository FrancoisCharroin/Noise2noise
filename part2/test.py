import torch.tensor
import math
import time

from modules import *


# =========================== Helper function ===========================
def generate_disc_set(nb):
    points = torch.empty((nb, 2), dtype=torch.float32).uniform_()
    labels = torch.le((points[:, 0] ** 2 + points[:, 1] ** 2).sqrt(), math.sqrt(2 / math.pi))
    return points, labels.to(torch.float32)


# =========================== Training functions ===========================
def train_model(model, train_input, train_target,
                loss_criterion="MSE", momentum=0., tar_scale=1.):
    nb_epochs = 100  # 25
    mini_batch_size = 100

    if loss_criterion == "MSE":
        criterion = LossMSE()
        eta = 5e-1
        diff_loss = 1e-5
    elif loss_criterion == "CrossEntropy":
        criterion = LossBCE()
        eta = 1e-1
        diff_loss = 1e-3
    else:
        raise NotImplementedError

    errs = []
    losses = []
    n_reset = 0
    for e in range(nb_epochs):
        if e % 50 == 0:
            start = time.time()
        nb_errors = 0
        loss_inter = 0.
        model.zero_grad()

        for p in range(0, train_input.size(0), mini_batch_size):
            output = model.forward(train_input[p: p + mini_batch_size])
            tar = train_target[p: p + mini_batch_size].reshape(-1, 1)

            loss = criterion.forward(output, tar)
            loss_inter += loss.sum().item()

            grad_inter = criterion.backward(output, tar)
            model.backward(grad_inter)

            if p % mini_batch_size == 0 or p == train_input.size(0) - 1:
                model.sgd(eta / mini_batch_size, momentum=momentum)
                model.zero_grad()

            wrong_pred = torch.abs(output - tar) > tar_scale / 2.
            nb_errors += wrong_pred.sum()

        errs.append(nb_errors)
        losses.append(loss_inter)

        if e == 1 or (e + 1) % 50 == 0:
            print("Epoch {:d} (time {:0.2f}[s])\t train error: {:0.2%} \t loss: {:0.2f}"
                  .format(e + 1, (time.time() - start), nb_errors / train_input.size(0),
                          loss_inter))

        """If loss doesn't go down and the train error is ~50%, reset the params"""
        if e >= 10 and abs(losses[-2] - losses[-1]) < diff_loss:
            if errs[-1] / train_input.size(0) > 0.4:
                print("RESET PARAMS\n epochs {:d} \ttrain error {:.2%}\t diff_loss {:.8f}".
                      format(e, nb_errors/train_input.size(0), abs(losses[-2] - losses[-1])))
                model.reset_params()
                n_reset += 1
            else:
                pass

    return errs, losses, n_reset


def compute_nb_errors(model, test_input, test_target, tar_scale=1.):
    nb_data_errors = 0

    for p in range(0, test_input.size(0)):
        output = model.forward(test_input[p])
        wrong_pred = torch.abs(output - test_target[p]) > tar_scale / 2.
        if wrong_pred:
            nb_data_errors += 1

    return nb_data_errors


def run_miniproj2():
    torch.set_grad_enabled(False)
    n_samples = 1000
    n_hidden = 25
    tar_scale = 0.9
    nb_crossval_iter = 10

    loss_criterion = "CrossEntropy"  # "MSE"
    momentum = 0.2  

    nb_test_errors = torch.zeros(nb_crossval_iter)
    n_reset = 0

    for i in range(nb_crossval_iter):
        print("\nCrossval number: {:d}".format(i))
        train_input, train_target = generate_disc_set(n_samples)
        test_input, test_target = generate_disc_set(n_samples)

        train_target *= tar_scale  # to stay in the tanh range
        train_input.sub_(train_input.mean())
        train_input.div_(train_input.std())

        test_input.sub_(test_input.mean())
        test_input.div_(test_input.std())

        model = Sequential(Linear(2, n_hidden, "elu"), ELU(),
                           Linear(n_hidden, n_hidden, "relu"), ReLU(),
                           Linear(n_hidden, n_hidden, "selu"), SeLU(),
                           Linear(n_hidden, 1, "sigmoid"), Sigmoid())

        error, loss, tmp = train_model(model, train_input, train_target, loss_criterion,
                                       momentum, tar_scale)
        n_reset += tmp

        nb_test_errors[i] = compute_nb_errors(model, test_input, test_target, tar_scale)
        print("Test error {:0.2%}".format(nb_test_errors[i] / test_input.size(0)))

    print("\n******\n+++ Result of cross validation +++\n"
          "{:d} time(s) parameter reset\n"
          "Test error:\t mean {:0.2%}\t std {:0.2%}\n*****".
          format(n_reset,
                 nb_test_errors.mean() / test_input.size(0),
                 nb_test_errors.std() / test_input.size(0)))


run_miniproj2()
