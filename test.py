import os

from sklearn.model_selection import train_test_split

import data_wrangler
import model
import train




def main():

    X, y = data_wrangler.read_prostate_cancer_dataset(
        os.path.expanduser('~/Downloads/Prostate_cancer_dataset.zip')
    )
    X, Xt, y, yt = train_test_split(X, y)

    net = model.UVAENet(X[0].shape,
                        encoder_config={'input_channels': 3},
                        vae_config={'initial_channels': 4},
                        semantic_config={'output_channels': 2})
    train.train(net, X, y, epochs=1, batch_size=4)


if __name__ == '__main__':
    main()
