"""
The :mod:`datascience-snippets.iterators module includes iterators that
can be used to combine samples in multiple ways.

These are particulary useful when using scikit-learn's hyperparameter
optimization algorithms. To use them in this context, just pass an instance
of the desired iterator to the 'cv' parameter of
e.g. sklearn.grid_search.GridSearchCV.

"""
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

class NRepeatedKFold(object):
    """K-Folds cross validation reiterated n times

    This is a repeating variant of scikit-learn's KFold (and StratifiedKFold) class.

    Instead of generating an iterator for a single K-folds cross validation,
    this class iterates over n such cross validations, where the split is randomly
    choosen for each single CV.

    Parameters
    ----------
    y : array-like, [n_samples]
        List of samples to split.

    n_folds : int, default = 3
        Number of folds for each CV. Must be at least 2.

    repetitions : int, default=3
        Number of times the cross validation should be repeated. Must be at
        least 1.

    stratify : Boolean, default=False
        If True, use sklearn.module_selection.StratifiedKFold instead of sklearn.module_selection.KFold

    shuffle : Boolean, default=False
        If True, shuffle data before folding.

    random_state : Int or None
        When shuffle=True, pseudo-random number generator state used for shuffling. If None,
        use default numpy RNG for shuffling.
        Note: Other than in sklearn's KFold methods, this parameter can not be an instance of numpy.random.RandomState.
    """

    def __init__(self, y, n_folds=3, repetitions=3, stratify=False, shuffle=False,
                 random_state=None):

        self.n_folds = n_folds
        self.repetitions = repetitions
        self.curRep = 0
        self.cross_validations = []

        if stratify:
            kfold_method = StratifiedKFold
        else:
            kfold_method = KFold

        for i in range(repetitions):
            if isinstance(random_state, int):
                # give every cross validation cycle a unique random_state, so they are different from one another
                self.cross_validations.append(kfold_method(y, n_folds, shuffle, random_state + i))
            else:
                self.cross_validations.append(kfold_method(y, n_folds, shuffle))

    def __iter__(self):
        self.curRep = 0
        self.cv = iter(self.cross_validations[0])
        return self

    def __len__(self):
        return self.repetitions * self.n_folds

    def __next__(self):
        try:
            iterator_element = next(self.cv)
        except StopIteration:
            self.curRep += 1
            if self.curRep == self.repetitions:
                raise StopIteration
            else:
                self.cv = iter(self.cross_validations[self.curRep])
                iterator_element = next(self.cv)
        return iterator_element
