import torch
import os
import shutil
import time


def load_model(net, model):
    """
    Load pretrained model from saved_model

    """
    assert os.path.exists(model), 'Error: no pretrained model found!'
    pre_trained_model = torch.load(model)
    net_param = pre_trained_model['state_dict']
    print('        pretrained best_acc1: {:.4f} '.format(pre_trained_model['best_acc1']))
    net.load_state_dict(net_param)

    return net


def validate(val_loader, model, criterion, args, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1(%) {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5(%) {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def validate_train(val_loader, model, criterion, args, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1(%) {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5(%) {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
    return top1.avg


def validate_with_outputs(val_loader, model, Emodel, criterion, args):

    # switch to evaluate mode
    model.eval()
    Emodel.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            # compute output
            output = model(input)
            eoutput = Emodel(input)

            loss = criterion(output, target)
            eloss = criterion(eoutput, target)
            assert (torch.allclose(loss, eloss) is True)
            # assert ((torch.eq(loss, eloss)).all() == 1)

            _, pred = output.topk(1, 1, True, True)
            _, epred = eoutput.topk(1, 1, True, True)

            assert (torch.allclose(pred, epred) is True)

            assert (torch.allclose(output, eoutput) is True)
            # assert ((torch.eq(output, eoutput)).all() == 1)


def dummy_validate(inputs, model, args):

    # switch to evaluate mode
    model.eval()
    model.cuda(args.gpu)
    with torch.no_grad():
        inputs = inputs.cuda(args.gpu, non_blocking=True)
        outputs = model(inputs)

    return outputs


def dummy_validate_with_outputs(inputs, model, Emodel, args):

    # switch to evaluate mode
    model.eval()
    Emodel.eval()
    model.cuda(args.gpu)
    Emodel.cuda(args.gpu)

    with torch.no_grad():
        if args.gpu is not None:
            inputs = inputs.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(inputs)
        eoutput = Emodel(inputs)

        _, pred = output.topk(1, 1, True, True)
        _, epred = eoutput.topk(1, 1, True, True)

        assert (torch.allclose(pred, epred) is True)
        assert (torch.allclose(output, eoutput) is True)


def save_checkpoint(state, is_best, filepath, filename='checkpoint.pth.tar'):
    torch.save(state, filepath + filename)
    if is_best:
        shutil.copyfile(filepath + filename, filepath + 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.epochs < 100:
        lr = args.lr * (0.1 ** (epoch // 30))
    else:
        if epoch < 201:
            lr = args.lr * (0.1 ** (epoch // 200))
        else:
            if epoch < 301:
                lr = args.lr * (0.1 ** (epoch // 200))
            else:
                lr = args.lr * (0.1 ** (epoch // 300))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res










