import build_kernel as bk
import torch


class AlternatingOptimizer:
    def __init__(self, opts):
        self.image_model   = bk.KernelMethodY(opts,'image')
        self.text_model    = bk.KernelMethodY(opts,'text')


    def main(self, X_I, Y_I, S_I, Y_D, S_D, text_embeddings, num_iters, get_zeroshot_predictions, cfg):

        self.image_model.solver(X = X_I, Y = Y_I, S = S_I, Z = None)

        y_binary = ((Y_I + 1) / 2)[:, 1].int()
        X_T = text_embeddings[y_binary]

        for iter in range(num_iters):

            # Updating the pseudo-labels
            if iter > 0:
                debias_image_train, debias_text_train = self.get_feat(X_I, X_T)
                text_embeddings_debias = self.get_textfeat(text_embeddings)
                dataset_predictions_train = get_zeroshot_predictions(debias_image_train,
                                                                     text_embeddings_debias,
                                                                     cfg,
                                                                     temperature=100.)
                Y_D = (torch.nn.functional.one_hot(torch.from_numpy(dataset_predictions_train.astype(int)),
                                                              num_classes=2)) * 2 - 1
                Y_I = Y_D

                y_binary = ((Y_I + 1) / 2)[:, 1].int()
                X_T = text_embeddings[y_binary]
                


            Z_I = self.image_model.encod(X_I)


            self.text_model.solver(X=X_T, Y=Y_D, S=S_D, Z=Z_I)


            Z_D = self.text_model.encod(X_T)

            
            self.image_model.solver(X=X_I, Y=Y_I, S=S_I, Z=Z_D)

            print(f'Training {iter+1}/{num_iters} done!')


    def get_feat(self, X_I, X_D):

        Z_D = self.text_model.encod(X_D)
        Z_I = self.image_model.encod(X_I)

        return Z_I, Z_D

    def get_textfeat(self, X_D):

        Z_D = self.text_model.encod(X_D)

        return Z_D
