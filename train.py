import numpy as np
import torch
from network import load_base_model
import argparse
from clip import clip
from defaults import _C as cfg
from network.clip import evaluate_clip
from base import AlternatingOptimizer
from datasets import Clipfeature
from torch.utils.data import DataLoader

def main():

    parser = argparse.ArgumentParser(description="fairerclip")
    parser.add_argument(
        "--config-file",
        default="configs/debias_waterbird.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )

    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line",
        default=None,
        nargs='+',
    )

    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    if args.opts is not None:
        cfg.merge_from_list(args.opts)
    cfg.num_workers=2
    random_seed = cfg.seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    print('tau_i: %.2f tau_z_i: %.2f tau_t: %.2f tau_z_t: %.2f'%(cfg.tau_i,cfg.tau_z_i,cfg.tau_t,cfg.tau_z_t))

    # ----------------------------------------------------------
    # loading train / test sets
    # ----------------------------------------------------------
    trainset = Clipfeature('train',cfg)
    traindata = DataLoader(trainset,batch_size=trainset.__len__(),shuffle=False)

    testset = Clipfeature('test', cfg)
    testdata = DataLoader(testset, batch_size=testset.__len__(), shuffle=False)

    # ----------------------------------------------------------
    # loading model /label for zero-shot testing
    # ----------------------------------------------------------
    base_model_args = cfg.load_base_model.split('_')
    base_model_components = load_base_model(base_model_args, cfg, clip=clip)
    base_model, base_transform, get_embeddings, get_dataset_embeddings, get_zeroshot_predictions = base_model_components

    if cfg.dataset=='waterbirds':
        text_descriptions = ['This is a picture of a landbird.', 'This is a picture of a waterbird.']
    else:
        text_descriptions = ['A photo of a celebrity with dark hair.', 'A photo of a celebrity with blond hair.']


    query_embeddings = get_embeddings(text_descriptions,
                                      base_model,
                                      cfg,
                                      normalize=True,
                                      verbose=False)
    text_embeddings = query_embeddings.float().to(cfg.device)


    for i, (imfeat_train, textfeat_train, labels_train_y, labels_train_s, labels_train_y_gt) in enumerate(traindata):
        if cfg.nolabel == True:
            iter = cfg.iters
        else:
            iter = 1
        model = AlternatingOptimizer(cfg)

        model.main(imfeat_train, labels_train_y, labels_train_s, labels_train_y, labels_train_s, text_embeddings, iter, get_zeroshot_predictions, cfg)


        for i, (imfeat_test, textfeat_test, labels_test_y, labels_test_s, labels_test_y_gt) in enumerate(testdata):
            debias_image, debias_text = model.get_feat(imfeat_test, textfeat_test)

            text_embeddings_debias = model.get_textfeat(text_embeddings)

            dataset_predictions = get_zeroshot_predictions(debias_image,
                                                              text_embeddings_debias,
                                                              cfg,
                                                              temperature=100.)

            print('result for testing set:')
            avg_acc, robust_acc, groups_acc = evaluate_clip(dataset_predictions,
                                                            labels_test_y_gt, 
                                                            labels_test_s,
                                                            verbose=True)

if __name__ == "__main__":
    main()