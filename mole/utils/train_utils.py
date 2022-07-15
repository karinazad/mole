
def early_stop(loss, prev_val_loss, n_no_improvement):
    if loss >= prev_val_loss:
        n_no_improvement += 1
    else:
        n_no_improvement = 0
        prev_val_loss = loss

    return prev_val_loss, n_no_improvement

