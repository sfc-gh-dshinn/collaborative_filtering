"""
the :mod:`matrix_factorization` module includes some algorithms using matrix
factorization.
"""

import numbers
import numpy as np
import pandas as pd


class SVD():
    """The famous *SVD* algorithm, as popularized by `Simon Funk
    <https://sifter.org/~simon/journal/20061211.html>`_ during the Netflix
    Prize. When baselines are not used, this is equivalent to Probabilistic
    Matrix Factorization :cite:`salakhutdinov2008a` (see :ref:`note
    <unbiased_note>` below).

    The prediction :math:`\\hat{r}_{ui}` is set as:

    .. math::
        \\hat{r}_{ui} = \\mu + b_u + b_i + q_i^Tp_u

    If user :math:`u` is unknown, then the bias :math:`b_u` and the factors
    :math:`p_u` are assumed to be zero. The same applies for item :math:`i`
    with :math:`b_i` and :math:`q_i`.

    For details, see equation (5) from :cite:`Koren:2009`. See also
    :cite:`Ricci:2010`, section 5.3.1.

    To estimate all the unknown, we minimize the following regularized squared
    error:

    .. math::
        \\sum_{r_{ui} \\in R_{train}} \\left(r_{ui} - \\hat{r}_{ui} \\right)^2 +
        \\lambda\\left(b_i^2 + b_u^2 + ||q_i||^2 + ||p_u||^2\\right)


    The minimization is performed by a very straightforward stochastic gradient
    descent:

    .. math::
        b_u &\\leftarrow b_u &+ \\gamma (e_{ui} - \\lambda b_u)\\\\
        b_i &\\leftarrow b_i &+ \\gamma (e_{ui} - \\lambda b_i)\\\\
        p_u &\\leftarrow p_u &+ \\gamma (e_{ui} \\cdot q_i - \\lambda p_u)\\\\
        q_i &\\leftarrow q_i &+ \\gamma (e_{ui} \\cdot p_u - \\lambda q_i)

    where :math:`e_{ui} = r_{ui} - \\hat{r}_{ui}`. These steps are performed
    over all the ratings of the trainset and repeated ``n_epochs`` times.
    Baselines are initialized to ``0``. User and item factors are randomly
    initialized according to a normal distribution, which can be tuned using
    the ``init_mean`` and ``init_std_dev`` parameters.

    You also have control over the learning rate :math:`\\gamma` and the
    regularization term :math:`\\lambda`. Both can be different for each
    kind of parameter (see below). By default, learning rates are set to
    ``0.005`` and regularization terms are set to ``0.02``.

    .. _unbiased_note:

    .. note::
        You can choose to use an unbiased version of this algorithm, simply
        predicting:

        .. math::
            \\hat{r}_{ui} = q_i^Tp_u

        This is equivalent to Probabilistic Matrix Factorization
        (:cite:`salakhutdinov2008a`, section 2) and can be achieved by setting
        the ``biased`` parameter to ``False``.


    Args:
        n_factors: The number of factors. Default is ``100``.
        n_epochs: The number of iteration of the SGD procedure. Default is
            ``20``.
        biased(bool): Whether to use baselines (or biases). See :ref:`note
            <unbiased_note>` above.  Default is ``True``.
        init_mean: The mean of the normal distribution for factor vectors
            initialization. Default is ``0``.
        init_std_dev: The standard deviation of the normal distribution for
            factor vectors initialization. Default is ``0.1``.
        lr_all: The learning rate for all parameters. Default is ``0.005``.
        reg_all: The regularization term for all parameters. Default is
            ``0.02``.
        lr_bu: The learning rate for :math:`b_u`. Takes precedence over
            ``lr_all`` if set. Default is ``None``.
        lr_bi: The learning rate for :math:`b_i`. Takes precedence over
            ``lr_all`` if set. Default is ``None``.
        lr_pu: The learning rate for :math:`p_u`. Takes precedence over
            ``lr_all`` if set. Default is ``None``.
        lr_qi: The learning rate for :math:`q_i`. Takes precedence over
            ``lr_all`` if set. Default is ``None``.
        reg_bu: The regularization term for :math:`b_u`. Takes precedence
            over ``reg_all`` if set. Default is ``None``.
        reg_bi: The regularization term for :math:`b_i`. Takes precedence
            over ``reg_all`` if set. Default is ``None``.
        reg_pu: The regularization term for :math:`p_u`. Takes precedence
            over ``reg_all`` if set. Default is ``None``.
        reg_qi: The regularization term for :math:`q_i`. Takes precedence
            over ``reg_all`` if set. Default is ``None``.
        random_state(int, RandomState instance from numpy, or ``None``):
            Determines the RNG that will be used for initialization. If
            int, ``random_state`` will be used as a seed for a new RNG. This is
            useful to get the same initialization over multiple calls to
            ``fit()``.  If RandomState instance, this same instance is used as
            RNG. If ``None``, the current RNG from numpy is used.  Default is
            ``None``.
        verbose: If ``True``, prints the current epoch. Default is ``False``.

    Attributes (+1 is empty to capture default return of 0s):
        pu(numpy array of size (n_users+1, n_factors)): The user factors (only
            exists if ``fit()`` has been called)
        qi(numpy array of size (n_items+1, n_factors)): The item factors (only
            exists if ``fit()`` has been called)
        bu(numpy array of size (n_users+1)): The user biases (only
            exists if ``fit()`` has been called)
        bi(numpy array of size (n_items+1)): The item biases (only
            exists if ``fit()`` has been called)
    """

    def __init__(self, n_factors=100, n_epochs=20, biased=True, init_mean=0,
                 init_std_dev=.1, lr_all=.005,
                 reg_all=.02, lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None,
                 reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None,
                 random_state=None, verbose=False, shuffle=True):

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.biased = biased
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.lr_bu = lr_bu if lr_bu is not None else lr_all
        self.lr_bi = lr_bi if lr_bi is not None else lr_all
        self.lr_pu = lr_pu if lr_pu is not None else lr_all
        self.lr_qi = lr_qi if lr_qi is not None else lr_all
        self.reg_bu = reg_bu if reg_bu is not None else reg_all
        self.reg_bi = reg_bi if reg_bi is not None else reg_all
        self.reg_pu = reg_pu if reg_pu is not None else reg_all
        self.reg_qi = reg_qi if reg_qi is not None else reg_all
        self.random_state = random_state
        self.verbose = verbose
        self.shuffle = shuffle

    def fit(self, X, y):
        # OK, let's breath. I've seen so many different implementation of this
        # algorithm that I just not sure anymore of what it should do. I've
        # implemented the version as described in the BellKor papers (RS
        # Handbook, etc.). Mymedialite also does it this way. In his post
        # however, Funk seems to implicitly say that the algo looks like this
        # (see reg below):
        # for f in range(n_factors):
        #       for _ in range(n_iter):
        #           for u, i, r in all_ratings:
        #               err = r_ui - <p[u, :f+1], q[i, :f+1]>
        #               update p[u, f]
        #               update q[i, f]
        # which is also the way https://github.com/aaw/IncrementalSVD.jl
        # implemented it.
        #
        # Funk: "Anyway, this will train one feature (aspect), and in
        # particular will find the most prominent feature remaining (the one
        # that will most reduce the error that's left over after previously
        # trained features have done their best). When it's as good as it's
        # going to get, shift it onto the pile of done features, and start a
        # new one. For efficiency's sake, cache the residuals (all 100 million
        # of them) so when you're training feature 72 you don't have to wait
        # for predictRating() to re-compute the contributions of the previous
        # 71 features. You will need 2 Gig of ram, a C compiler, and good
        # programming habits to do this."

        # A note on cythonization: I haven't dived into the details, but
        # accessing 2D arrays like pu using just one of the indices like pu[u]
        # is not efficient. That's why the old (cleaner) version can't be used
        # anymore, we need to compute the dot products by hand, and update
        # user and items factors by iterating over all factors...

        # Data validation and change to numpy.ndarray if pandas
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError("X must be pd.DataFrame or np.ndarray)")

        if isinstance(X, pd.DataFrame):
            X = X.values

        if X.shape[1] != 2:
            raise ValueError("X must have 2 columns")

        if not isinstance(y, (pd.Series, np.ndarray)):
            raise TypeError("y must be pd.Series or np.ndarray)")

        if isinstance(y, pd.Series):
            y = y.values

        # Make sure y only has single dimension
        if len(y.shape) != 1:
            raise ValueError("y must be a single dimension array .shape == (n,)")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same length")

        # New calculations since removing dependence on `trainset`
        self.global_mean = y.mean()
        self.mapping_user = {user_id: x for x, user_id in enumerate(np.unique(X[:, 0]))}
        self.mapping_item = {item_id: x for x, item_id in enumerate(np.unique(X[:, 1]))}
        self.n_users = len(self.mapping_user)
        self.n_items = len(self.mapping_item)

        rng = get_rng(self.random_state)

        # Note: last index for all factors are 0.0 for easy default for new users or items
        # user biases, last user used for default if new
        bu = np.zeros(self.n_users + 1, dtype=np.double)
        # item biases, last item used for default if new
        bi = np.zeros(self.n_items + 1, dtype=np.double)
        # user factors, last user used for default if new
        pu = rng.normal(self.init_mean, self.init_std_dev, size=(self.n_users + 1, self.n_factors))
        pu[-1] = 0.0
        # item factors, last item used for default if new
        qi = rng.normal(self.init_mean, self.init_std_dev, size=(self.n_items + 1, self.n_factors))
        qi[-1] = 0.0

        n_factors = self.n_factors
        biased = self.biased

        global_mean = self.global_mean

        lr_bu = self.lr_bu
        lr_bi = self.lr_bi
        lr_pu = self.lr_pu
        lr_qi = self.lr_qi

        reg_bu = self.reg_bu
        reg_bi = self.reg_bi
        reg_pu = self.reg_pu
        reg_qi = self.reg_qi

        # Create integer replacements for user and items, shuffle if True
        array_users = pd.Series(X[:, 0]).map(self.mapping_user).values
        array_items = pd.Series(X[:, 1]).map(self.mapping_item).values
        array_ratings = y

        if self.shuffle:
            shuffle_order = np.random.choice(range(X.shape[0]), X.shape[0], replace=False)
            array_users = array_users[shuffle_order]
            array_items = array_items[shuffle_order]
            array_ratings = y[shuffle_order]

        if not biased:
            global_mean = 0

        for current_epoch in range(self.n_epochs):
            if self.verbose:
                print("Processing epoch {}".format(current_epoch))

            for n_index in range(X.shape[0]):
                u = array_users[n_index]
                i = array_items[n_index]
                r = array_ratings[n_index]

                # compute current error
                dot = 0  # <q_i, p_u>
                for f in range(n_factors):
                    dot += qi[i, f] * pu[u, f]
                err = r - (global_mean + bu[u] + bi[i] + dot)

                # update biases
                if biased:
                    bu[u] += lr_bu * (err - reg_bu * bu[u])
                    bi[i] += lr_bi * (err - reg_bi * bi[i])

                # update factors
                for f in range(n_factors):
                    puf = pu[u, f]
                    qif = qi[i, f]
                    pu[u, f] += lr_pu * (err * qif - reg_pu * puf)
                    qi[i, f] += lr_qi * (err * puf - reg_qi * qif)

        self.bu = np.asarray(bu)
        self.bi = np.asarray(bi)
        self.pu = np.asarray(pu)
        self.qi = np.asarray(qi)
    

    def predict(self, X):
        # Data validation and change to numpy.ndarray if pandas
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError("X must be pd.DataFrame or np.ndarray)")

        if isinstance(X, pd.DataFrame):
            X = X.values

        if X.shape[1] != 2:
            raise ValueError("X must have 2 columns")

        # Note that the last index represents missing and will return 0s
        array_users = np.asarray([self.mapping_user.get(x, self.n_users) for x in X[:, 0]])
        array_items = np.asarray([self.mapping_item.get(x, self.n_items) for x in X[:, 1]])

        bus = self.bu[array_users]
        bis = self.bi[array_items]

        pus = self.pu[array_users]
        qis = self.qi[array_items]

        est = bus + bis + (pus * qis).sum(1)

        if self.biased:
            est +=  self.global_mean

        return est


def get_rng(random_state):
    """Return a 'validated' RNG.

    If random_state is None, use RandomState singleton from numpy.  Else if
    it's an integer, consider it's a seed and initialized an rng with that
    seed. If it's already an rng, return it.
    """
    if random_state is None:
        return np.random.mtrand._rand
    elif isinstance(random_state, (numbers.Integral, np.integer)):
        return np.random.RandomState(random_state)
    if isinstance(random_state, np.random.RandomState):
        return random_state
    raise ValueError(
        "Wrong random state. Expecting None, an int or a numpy "
        "RandomState instance, got a "
        "{}".format(type(random_state))
    )