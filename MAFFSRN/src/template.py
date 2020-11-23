def set_template(args):
    # Set the templates here
    if args.template.find('jpeg') >= 0:
        args.data_train = 'DIV2K_jpeg'
        args.data_test = 'DIV2K_jpeg'
        args.epochs = 200
        args.decay = '100'

    if args.template.find('MAFFSRN') >= 0:
        args.model = 'MAFFSRN'
        args.n_FFGs = 4
        args.n_feats = 32
        

  
    if args.template.find('MAFFSRN-L') >= 0:
        args.model = 'MAFFSRN'
        args.n_FFGs = 8
        args.n_feats = 32
        

  
