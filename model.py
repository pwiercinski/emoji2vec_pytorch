import torch
from torch import optim, autograd
import gensim.models as gsm
from pytorch_network import Net
import numpy as np
from naga.shared.kb import BatchNegSampler
from sklearn import metrics


class ModelParams:
    """Convenience class for passing around model parameters"""

    def __init__(self, in_dim, out_dim, max_epochs, pos_ex, neg_ratio, learning_rate, dropout, class_threshold):
        """Create a struct of all parameters that get fed into the model

        Args:
            in_dim: Dimension of the word vectors supplied to the algorithm (i.e. word2vec)
            out_dim: Dimension of the output emoji vectors of the algorithm
            pos_ex: Number of positive examples per batch
            max_epochs: Max number of training epochs
            neg_ratio: Ratio between negative examples and positive examples in a batch
            learning_rate: Learning rate
            dropout: Dropout rate
            class_threshold: Classification threshold for accuracy
        """
        self.class_threshold = class_threshold
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.neg_ratio = neg_ratio
        self.max_epochs = max_epochs
        self.pos_ex = pos_ex
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.mb = self.pos_ex * (1 + self.neg_ratio)

    def model_folder(self, dataset_name):
        """Get the model path for a given dataset

        Args:
            dataset_name: The name of the dataset we used to generate training data

        Returns:
            The model path for a given dataset
        """
        return str.format('./results/{}/k-{}_pos-{}_rat-{}_ep-{}_dr-{}', dataset_name, self.out_dim, self.pos_ex,
                       self.neg_ratio, self.max_epochs, int(self.dropout * 10))

        
class Emoji2Vec:

	def __init__(self, model_params, num_emojis, embeddings_array):
		self.params = model_params
		self.num_emojis = num_emojis
		self.embeddings_array = embeddings_array
		self.nn = Net(input_size=self.params.in_dim, output_size=self.params.out_dim, num_emojis=num_emojis,
						dropout=self.params.dropout)

	def train(self, kb, epochs, learning_rate):
		batcher = BatchNegSampler(kb=kb, arity=1, batch_size=self.params.mb, neg_per_pos=self.params.neg_ratio)
		opt = optim.Adam(params=self.nn.parameters(), lr=learning_rate)
		criterion = torch.nn.BCELoss()
		for epoch in range(epochs):
			epoch_loss = []
			epoch_acc = []
			epoch_f1 = []
			for batch in batcher:
				emoj_idxs = batch[0]
				phr_idxs = batch[1]
				labels = autograd.Variable(torch.FloatTensor(batch[2]))
				input = autograd.Variable(torch.tensor(self.embeddings_array[phr_idxs]))
				out = self.nn(input, emoj_idxs)
				loss = criterion(out, labels)
				batch_acc = torch.div((labels == torch.round(out)).sum().double(), batcher.batch_size)
				batch_f1 = metrics.f1_score(labels.detach().numpy(), torch.round(out).detach().numpy())
				self.nn.zero_grad()
				epoch_loss.append(np.float32(loss.data))
				epoch_acc.append(batch_acc)
				epoch_f1.append(batch_f1)
				loss.backward()
				opt.step()
			epoch_loss = np.round(np.mean(epoch_loss), 2)
			epoch_acc = np.round(np.mean(epoch_acc), 2)
			epoch_f1 = np.round(np.mean(epoch_f1), 2)
			print(str.format('Epoch: {} \n Training loss: {} \n Training acc: {} \n Training f1: {} \n ===================',
								epoch + 1, str(epoch_loss), epoch_acc, epoch_f1))

	def predict(self, dset, threshold):
		"""Generate predictions on a given set of examples using TensorFlow

		Args:
			dset: Dataset tuple (emoji_ix, phrase_ix, truth)
			threshold: Threshold for classification

		Returns:
			Returns predicted values for an example, as well as the true value
		"""
		phr_ix, em_ix, truth = dset
		res = self.nn(x=torch.FloatTensor(self.embeddings_array[phr_ix]), emoji_ids=em_ix)
		y_pred = np.asarray([1 if y > threshold else 0 for y in res])
		y_true = np.asarray(truth).astype(int)

		return y_pred, y_true

	def accuracy(self, dset, threshold):
		"""Calculate the accuracy of a dataset at a given threshold.

		Args:
			dset: Dataset tuple (emoji_ix, phrase_ix, truth)
			threshold: Threshold for classification

		Returns:
			Accuracy
		"""
		y_pred, y_true = self.predict(dset, threshold)

		return metrics.accuracy_score(y_true=y_true, y_pred=y_pred)

	def f1_score(self, dset, threshold=0.5):
		"""Calculate the f1 score of a dataset at a given classification threshold.

		Args:
			dset: Dataset tuple (emoji_ix, phrase_ix, truth)
			threshold: Threshold for classification

		Returns:
			F1 score
		"""
		y_pred, y_true = self.predict(dset, threshold)

		return metrics.f1_score(y_true=y_true, y_pred=y_pred)

	def auc(self, dset):
		"""Calculates the Area under the Curve for the f1 at various thresholds

		Args:
			dset: Dataset tuple (emoji_ix, phrase_ix, truth)

		Returns:

		"""
		phr_ix, em_ix, truth = dset
		res = self.nn(x=torch.FloatTensor(self.embeddings_array[phr_ix]), emoji_ids=em_ix)
		y_true = np.asarray(truth).astype(int)

		return metrics.roc_auc_score(y_true, res.detach().numpy())

	def roc_vals(self, dset):
		"""Generates a receiver operating curve for the dataset

		Args:
			dset: Dataset tuple (emoji_ix, phrase_ix, truth)

		Returns:
			Points on the curve
		"""
		phr_ix, em_ix, truth = dset
		res = self.nn(x=torch.FloatTensor(self.embeddings_array[phr_ix]), emoji_ids=em_ix)
		y_true = np.asarray(truth).astype(int)

		return metrics.roc_curve(y_true, res.detach().numpy())

	def create_gensim_files(self, model_folder, ind2emoj, out_dim=300):
		"""Given a trained session and a destination path (model_folder), generate the gensim binaries
		for a model.

		Args:
			model_folder: Folder in which to generate the files
			ind2emoj: Mapping from indices to emoji
			out_dim: Output dimension of the emoji vectors

		Returns:
		"""

		vecs = list(self.nn.parameters())[0]
		txt_path = model_folder + '/test_emoji2vec.txt'
		bin_path = model_folder + '/test_emoji2vec.bin'
		f = open(txt_path, 'w', encoding="utf8")
		f.write('%d %d\n' % (len(vecs), out_dim))
		for i in range(len(vecs)):
			f.write(ind2emoj[i] + ' ')
			for j in range(out_dim):
				f.write(str.format('{} ', vecs[i][j]))
			f.write('\n')
		f.close()

		e2v = gsm.KeyedVectors.load_word2vec_format(txt_path, binary=False)
		e2v.save_word2vec_format(bin_path, binary=True)

		return e2v
