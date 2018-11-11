# External dependencies
import os
import pickle as pk

# Internal dependencies
from parameter_parser import CliParser
from model import Emoji2Vec
import torch
from utils import build_kb, get_examples_from_kb, generate_embeddings, get_metrics, generate_predictions


# Execute training sequence
def __run_training():
    # Setup
    args = CliParser()
    args.print_params('EMOJI TRAINING')

    # Build knowledge base
    print('reading training data from: ' + args.data_folder)
    kb, ind2phr, ind2emoji = build_kb(args.data_folder)

    # Save the mapping from index to emoji
    pk.dump(ind2emoji, open(args.mapping_file, 'wb'))

    # Get the embeddings for each phrase in the training set
    embeddings_array = generate_embeddings(ind2phr=ind2phr, kb=kb, embeddings_file=args.embeddings_file,
                                           word2vec_file=args.word2vec_file)

    # Get examples of each example type in two sets. This is just a reprocessing of the knowledge base for efficiency,
    # so we don't have to generate the train and dev set on each train
    train_set = get_examples_from_kb(kb=kb, example_type='train')
    dev_set = get_examples_from_kb(kb=kb, example_type='dev')

    train_save_evaluate(params=args.model_params, kb=kb, train_set=train_set, dev_set=dev_set,
                        ind2emoji=ind2emoji, embeddings_array=embeddings_array, dataset_name=args.dataset)


def train_save_evaluate(params, kb, train_set, dev_set, ind2emoji, embeddings_array, dataset_name):

    # If the minibatch is larger than the number of emojis we have, we can't fill train/test batches
    if params.mb > len(ind2emoji):
        print(str.format('Skipping: k={}, batch={}, epochs={}, ratio={}, dropout={}', params.out_dim,
                         params.pos_ex, params.max_epochs, params.neg_ratio, params.dropout))
        print("Can't have an mb > len(ind2emoji)")
        return "N/A"
    else:
        print(str.format('Training: k={}, batch={}, epochs={}, ratio={}, dropout={}', params.out_dim,
                         params.pos_ex, params.max_epochs, params.neg_ratio, params.dropout))

    model_folder = params.model_folder(dataset_name=dataset_name)
    model_path = model_folder + '/model.pt'
    
    dsets = {'train': train_set, 'dev': dev_set}
    predictions = dict()
    results = dict()

    if os.path.exists(model_path):
        predictions = pk.load(open(model_folder + '/results.p', 'rb'))

    else:

        model = Emoji2Vec(model_params=params, num_emojis=kb.dim_size(0), embeddings_array=embeddings_array)
        model.train(kb=kb, epochs=params.max_epochs, learning_rate=params.learning_rate)
        os.makedirs(model_folder)
        torch.save(model.nn, model_folder + '/model.pt')
        e2v = model.create_gensim_files(model_folder=model_folder, ind2emoj=ind2emoji, out_dim=params.out_dim)
        if params.in_dim != params.out_dim:
            embeddings_array = model.nn.project_embeddings(embeddings_array)
        for dset_name in dsets:
            _, pred_values, _, true_values = generate_predictions(e2v=e2v, dset=dsets[dset_name],
                                                                  phr_embeddings=embeddings_array,
                                                                  ind2emoji=ind2emoji,
                                                                  threshold=params.class_threshold)
            predictions[dset_name] = {
                'y_true': true_values,
                'y_pred': pred_values
            }

        pk.dump(predictions, open(model_folder + '/results.p', 'wb'))

    for dset_name in dsets:
        true_labels = [bool(x) for x in predictions[dset_name]['y_true']]
        pred_labels = [x >= params.class_threshold for x in predictions[dset_name]['y_pred']]
        true_values = predictions[dset_name]['y_true']
        pred_values = predictions[dset_name]['y_pred']
        # Calculate metrics
        acc, f1, auc = get_metrics(pred_labels, pred_values, true_labels, true_values)
        print(str.format('{}: Accuracy(>{}): {}, f1: {}, auc: {}', dset_name, params.class_threshold, acc, f1, auc))
        results[dset_name] = {
            'accuracy': acc,
            'f1': f1,
            'auc': auc
        }

    return results['dev']


if __name__ == '__main__':
    __run_training()
