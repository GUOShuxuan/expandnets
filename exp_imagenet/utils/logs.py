"""
logs.py: create log path, saving training logs
"""


def create_log_file_path(path, args):
    """
    create path to save logfiles according to experiments configurations

    :param path: root path saving logfiles
    :param args: name of experiment configuration [exp_id]
    :param param: parameters in experiments configurations [exp_id]
    :return:  the complete path where logfiles save
    """
    if args.resume is None:
        if args.pretrained is None:
            tags = '{}_epochs_{}_run_{}'.format(args.arch, str(args.epochs), str(args.run))
            log_path = path + '/' + tags + '/'
        else:
            tags = '{}_epochs_{}_run_{}'.format(args.arch, str(args.epochs), str(args.run)) + '_init_from_nonlinear'
            log_path = path + '/' + tags + '/'
    else:
        log_path = args.resume.rsplit('/', 1)[0] + '/'

    return log_path




