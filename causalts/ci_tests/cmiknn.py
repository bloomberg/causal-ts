"""
This file has been modified by Mohammad Fesanghary on July 28 2025.
Original Author: Jakob Runge <jakob@jakob-runge.com>
Original Package: tigramite
Original License: GNU General Public License v3.0
"""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import print_function

import warnings

import numpy as np
from numba import jit
from scipy import spatial, special

from .independence_tests_base import CondIndTest


class CMIknn(CondIndTest):
    r"""Conditional mutual information test based on nearest-neighbor estimator.

    Conditional mutual information is the most general dependency measure coming
    from an information-theoretic framework. It makes no assumptions about the
    parametric form of the dependencies by directly estimating the underlying
    joint density. The test here is based on the estimator in S. Frenzel and B.
    Pompe, Phys. Rev. Lett. 99, 204101 (2007), combined with a shuffle test to
    generate the distribution under the null hypothesis of independence first
    used in [Runge2018]_. The knn-estimator is suitable only for variables
    taking a continuous range of values. For discrete variables use the CMIsymb
    class.

    Notes
    -----
    CMI is given by

    .. math:: I(X;Y|Z) &= \int p(z)  \iint  p(x,y|z) \log
                \frac{ p(x,y |z)}{p(x|z)\cdot p(y |z)} \,dx dy dz

    Its knn-estimator is given by

    .. math:: \widehat{I}(X;Y|Z)  &=   \psi (k) + \frac{1}{T} \sum_{t=1}^T
            \left[ \psi(k_{Z,t}) - \psi(k_{XZ,t}) - \psi(k_{YZ,t}) \right]

    where :math:`\psi` is the Digamma function.  This estimator has as a
    parameter the number of nearest-neighbors :math:`k` which determines the
    size of hyper-cubes around each (high-dimensional) sample point. Then
    :math:`k_{Z,},k_{XZ},k_{YZ}` are the numbers of neighbors in the respective
    subspaces.

    :math:`k` can be viewed as a density smoothing parameter (although it is
    data-adaptive unlike fixed-bandwidth estimators). For large :math:`k`, the
    underlying dependencies are more smoothed and CMI has a larger bias,
    but lower variance, which is more important for significance testing. Note
    that the estimated CMI values can be slightly negative while CMI is a non-
    negative quantity.

    This method requires the scipy.spatial.cKDTree package.

    References
    ----------
    .. [Runge2018] J. Runge (2018): Conditional Independence Testing Based on
       a Nearest-Neighbor Estimator of Conditional Mutual Information.
       In Proceedings of the 21st International Conference on Artificial
       Intelligence and Statistics.
       http://proceedings.mlr.press/v84/runge18a.html

    Parameters
    ----------
    knn : int or float, optional
        Number of nearest-neighbors which determines the size of hyper-cubes
        around each (high-dimensional) sample point. If smaller than 1, this is
        computed as a fraction of T, hence knn=knn*T. For knn larger or equal to
        1, this is the absolute number. Default is 0.2.
    shuffle_neighbors : int, optional
        Number of nearest-neighbors within Z for the shuffle surrogates which
        determines the size of hyper-cubes around each (high-dimensional) sample
        point. Default is 5.
    transform : {'ranks', 'standardize', 'uniform', False}, optional
        Whether to transform the array beforehand by standardizing
        or transforming to uniform marginals. Default is 'ranks'.
    workers : int, optional
        Number of workers to use for parallel processing. If -1 is given
        all processors are used. Default is -1.
    model_selection_folds : int, optional
        Number of folds in cross-validation used in model selection.
        Default is 3.
    significance : str, optional
        Type of significance test to use. For CMIknn only 'fixed_thres' and
        'shuffle_test' are available. Default is 'shuffle_test'.
    **kwargs
        Arguments passed on to parent class CondIndTest.
    """

    @property
    def measure(self):
        """
        Concrete property to return the measure of the independence test
        """
        return self._measure

    def __init__(
        self,
        knn=0.2,
        shuffle_neighbors=5,
        significance="shuffle_test",
        transform="ranks",
        workers=-1,
        model_selection_folds=3,
        early_stop=True,
        target_quantile=0.05,
        **kwargs,
    ):
        # Set the member variables
        self.knn = knn
        self.shuffle_neighbors = shuffle_neighbors
        self.transform = transform
        self._measure = "cmi_knn"
        self.two_sided = False
        self.residual_based = False
        self.recycle_residuals = False
        self.workers = workers
        self.n_tests = 0
        self.n_actual_tests = 0
        self.early_stop = early_stop
        self.target_quantile = target_quantile
        self.model_selection_folds = model_selection_folds
        # Call the parent constructor
        CondIndTest.__init__(
            self, significance=significance, method="cmi_knn", **kwargs
        )
        self._measure = "cmi_knn"
        self.method = "cmi_knn"

        # Print some information about construction
        if self.verbosity > 0:
            if self.knn < 1:
                print(f"knn/T = {self.knn}")
            else:
                print(f"knn = {self.knn}")
            print(f"shuffle_neighbors = {self.shuffle_neighbors}\n")

    @jit(forceobj=True)
    def _get_nearest_neighbors(self, array, xyz, knn):
        """Returns nearest neighbors according to Frenzel and Pompe (2007).

        Retrieves the distances eps to the k-th nearest neighbors for every
        sample in joint space XYZ and returns the numbers of nearest neighbors
        within eps in subspaces Z, XZ, YZ.

        Parameters
        ----------
        array : array-like
            data array with X, Y, Z in rows and observations in columns

        xyz : array of ints
            XYZ identifier array of shape (dim,).

        knn : int or float
            Number of nearest-neighbors which determines the size of hyper-cubes
            around each (high-dimensional) sample point. If smaller than 1, this
            is computed as a fraction of T, hence knn=knn*T. For knn larger or
            equal to 1, this is the absolute number.

        Returns
        -------
        k_xz, k_yz, k_z : tuple of arrays of shape (T,)
            Nearest neighbors in subspaces.
        """

        array = array.astype(np.float64)
        xyz = xyz.astype(np.int32)

        dim, T = array.shape

        # Add noise to destroy ties...
        array += (
            1e-6
            * array.std(axis=1).reshape(dim, 1)
            * self.random_state.random((array.shape[0], array.shape[1]))
        )

        if self.transform == "standardize":
            # Standardize
            array = array.astype(np.float64)
            array -= array.mean(axis=1).reshape(dim, 1)
            std = array.std(axis=1)
            for i in range(dim):
                if std[i] != 0.0:
                    array[i] /= std[i]

            if np.any(std == 0.0) and self.verbosity > 0:
                warnings.warn("Possibly constant array!")

        elif self.transform == "uniform":
            array = self._trafo2uniform(array)
        elif self.transform == "ranks":
            array = array.argsort(axis=1).argsort(axis=1).astype(np.float64)

        array = array.T

        # Compute distance to k-th nearest neighbor, excluding points itself, hence k=[knn+1]
        tree_xyz = spatial.cKDTree(array)
        epsarray = tree_xyz.query(array, k=[knn + 1], p=np.inf, workers=self.workers)[
            0
        ][:, 0].astype(np.float64)
        # print("epsarray", epsarray)

        # To search neighbors < eps instead of <= eps, we need to reduce eps by a bit
        epsarray = np.multiply(epsarray, 0.999999999)
        # print("epsarray", epsarray)

        # Subsample indices
        x_indices = np.where(xyz == 0)[0]
        y_indices = np.where(xyz == 1)[0]
        z_indices = np.where(xyz == 2)[0]

        # Find nearest neighbors in subspaces
        xz = array[:, np.concatenate((x_indices, z_indices))]
        tree_xz = spatial.cKDTree(xz)
        k_xz = tree_xz.query_ball_point(
            xz, r=epsarray, p=np.inf, workers=self.workers, return_length=True
        )

        yz = array[:, np.concatenate((y_indices, z_indices))]
        tree_yz = spatial.cKDTree(yz)
        k_yz = tree_yz.query_ball_point(
            yz, r=epsarray, p=np.inf, workers=self.workers, return_length=True
        )

        if len(z_indices) > 0:
            z = array[:, z_indices]
            tree_z = spatial.cKDTree(z)
            k_z = tree_z.query_ball_point(
                z, r=epsarray, p=np.inf, workers=self.workers, return_length=True
            )
        else:
            # Number of neighbors is T when z is empty.
            k_z = np.full(T, T, dtype=np.float64)

        # print("knn", knn)
        # print("k_xz", k_xz)
        # print("k_yz", k_yz)
        # print("k_z", k_z)
        return k_xz, k_yz, k_z

    def get_dependence_measure(self, array, xyz, data_type=None):
        """Returns CMI estimate as described in Frenzel and Pompe PRL (2007).

        Parameters
        ----------
        array : array-like
            data array with X, Y, Z in rows and observations in columns

        xyz : array of ints
            XYZ identifier array of shape (dim,).

        Returns
        -------
        val : float
            Conditional mutual information estimate.
        """

        _, T = array.shape

        if self.knn < 1:
            knn_here = max(1, int(self.knn * T))
        else:
            knn_here = max(1, int(self.knn))

        k_xz, k_yz, k_z = self._get_nearest_neighbors(
            array=array, xyz=xyz, knn=knn_here
        )

        val = (
            special.digamma(knn_here)
            - (
                special.digamma(k_xz) + special.digamma(k_yz) - special.digamma(k_z)
            ).mean()
        )

        return val

    def get_shuffle_significance(
        self, array, xyz, value, return_null_dist=False, data_type=None
    ):
        """Returns p-value for nearest-neighbor shuffle significance test.

        For non-empty Z, overwrites get_shuffle_significance from the parent
        class  which is a block shuffle test, which does not preserve
        dependencies of X and Y with Z. Here the parameter shuffle_neighbors is
        used to permute only those values :math:`x_i` and :math:`x_j` for which
        :math:`z_j` is among the nearest niehgbors of :math:`z_i`. If Z is
        empty, the block-shuffle test is used.

        Parameters
        ----------
        array : array-like
            data array with X, Y, Z in rows and observations in columns

        xyz : array of ints
            XYZ identifier array of shape (dim,).

        value : number
            Value of test statistic for unshuffled estimate.

        Returns
        -------
        pval : float
            p-value
        """
        dim, T = array.shape

        # Skip shuffle test if value is above threshold
        # if value > self.minimum threshold:
        #     if return_null_dist:
        #         return 0., None
        #     else:
        #         return 0.

        # max_neighbors = max(1, int(max_neighbor_ratio*T))
        x_indices = np.where(xyz == 0)[0]
        z_indices = np.where(xyz == 2)[0]

        if len(z_indices) > 0 and self.shuffle_neighbors < T:
            if self.verbosity > 2:
                print(
                    f"Nearest-neighbor shuffle test with n={self.shuffle_neighbors} "
                    f"and {self.sig_samples} surrogates"
                )

            # Get nearest neighbors around each sample point in Z
            z_array = array[z_indices, :].copy()
            dim_z = len(z_indices)

            if self.transform == "standardize":
                # Standardize
                z_array = z_array.astype(np.float64)
                z_array -= z_array.mean(axis=1).reshape(dim_z, 1)
                std = z_array.std(axis=1)
                for i in range(dim_z):
                    if std[i] != 0.0:
                        z_array[i] /= std[i]

                if np.any(std == 0.0) and self.verbosity > 0:
                    warnings.warn("Possibly constant array!")
                    # raise ValueError("nans after standardizing, "
                    #                  "possibly constant array!")
            elif self.transform == "uniform":
                z_array = self._trafo2uniform(z_array)
            elif self.transform == "ranks":
                z_array = z_array.argsort(axis=1).argsort(axis=1).astype(np.float64)

            z_array = z_array.T

            tree_xyz = spatial.cKDTree(z_array)
            neighbors = tree_xyz.query(
                z_array, k=self.shuffle_neighbors, p=np.inf, eps=0.0
            )[1].astype(np.int32)

            null_dist = np.zeros(self.sig_samples)
            running_pvals = np.zeros(self.sig_samples)
            for sam in range(self.sig_samples):
                # Generate random order in which to go through indices loop in
                # next step
                order = self.random_state.permutation(T).astype(np.int32)

                # Shuffle neighbor indices for each sample index
                for i, neighbor in enumerate(neighbors):
                    self.random_state.shuffle(neighbor)
                # neighbors = self.random_state.permuted(neighbors, axis=1)

                # Select a series of neighbor indices that contains as few as
                # possible duplicates
                restricted_permutation = self.get_restricted_permutation(
                    T=T,
                    shuffle_neighbors=self.shuffle_neighbors,
                    neighbors=neighbors,
                    order=order,
                )

                array_shuffled = np.copy(array)
                for i in x_indices:
                    array_shuffled[i] = array[i, restricted_permutation]

                null_dist[sam] = self.get_dependence_measure(array_shuffled, xyz)

                if self.early_stop:
                    running_pvals[sam] = (null_dist[: (sam + 1)] >= value).mean()
                if (
                    self.early_stop
                    and sam > 20
                    and sam % 20 == 0
                    and (
                        running_pvals[sam] > (self.target_quantile + 0.1)
                        or np.std(running_pvals[:sam]) == 0
                    )
                ):
                    std_ = np.std(null_dist[:sam])
                    q_ = np.quantile(null_dist[:sam], self.target_quantile)
                    z_ = np.abs(value - q_) / std_
                    if z_ > 5:
                        break
                    std_pvals = np.std(running_pvals[:sam])
                    z_pval = (
                        np.median(running_pvals[:sam]) - self.target_quantile
                    ) / np.maximum(std_pvals, 0.0001)
                    if z_pval >= 3:
                        break
            null_dist = null_dist[:sam]
        else:
            null_dist = self._get_shuffle_dist(
                array,
                xyz,
                self.get_dependence_measure,
                sig_samples=self.sig_samples,
                sig_blocklength=self.sig_blocklength,
                verbosity=self.verbosity,
                statistics=value,
                early_stop=self.early_stop,
                target_quantile=self.target_quantile,
            )

        pval = (null_dist >= value).mean()

        if return_null_dist:
            # Sort
            null_dist.sort()
            return pval, null_dist
        return pval

    def get_conditional_entropy(self, array, xyz):
        """Returns the nearest-neighbor conditional entropy estimate of H(X|Y).

        Parameters
        ----------
        array : array-like
            data array with X, Y in rows and observations in columns

        xyz : array of ints
            XYZ identifier array of shape (dim,). Here only uses 0 for X and
            1 for Y.

        Returns
        -------
        val : float
            Entropy estimate.
        """

        dim, T = array.shape

        if self.knn < 1:
            knn_here = max(1, int(self.knn * T))
        else:
            knn_here = max(1, int(self.knn))

        array = array.astype(np.float64)

        # Add noise to destroy ties...
        array += (
            1e-6
            * array.std(axis=1).reshape(dim, 1)
            * self.random_state.random((array.shape[0], array.shape[1]))
        )

        if self.transform == "standardize":
            # Standardize
            array = array.astype(np.float64)
            array -= array.mean(axis=1).reshape(dim, 1)
            std = array.std(axis=1)
            for i in range(dim):
                if std[i] != 0.0:
                    array[i] /= std[i]

            if np.any(std == 0.0) and self.verbosity > 0:
                warnings.warn("Possibly constant array!")
            # if np.isnan(array).sum() != 0:
            #     raise ValueError("nans after standardizing, "
            #                      "possibly constant array!")
        elif self.transform == "uniform":
            array = self._trafo2uniform(array)
        elif self.transform == "ranks":
            array = array.argsort(axis=1).argsort(axis=1).astype(np.float64)

        # Compute conditional entropy as H(X|Y) = H(X) - I(X;Y)

        # First compute H(X)
        # Use cKDTree to get distances eps to the k-th nearest neighbors for
        # every sample in joint space X with maximum norm
        x_indices = np.where(xyz == 0)[0]
        y_indices = np.where(xyz == 1)[0]

        dim_x = int(np.where(xyz == 0)[0][-1] + 1)
        if 1 in xyz:
            dim_y = int(np.where(xyz == 1)[0][-1] + 1 - dim_x)
        else:
            dim_y = 0

        x_array = array[x_indices, :].T.copy()
        tree_xyz = spatial.cKDTree(x_array)
        epsarray = tree_xyz.query(
            x_array, k=[knn_here + 1], p=np.inf, eps=0.0, workers=self.workers
        )[0][:, 0].astype(np.float64)

        h_x = (
            -special.digamma(knn_here)
            + special.digamma(T)
            + dim_x * np.log(2.0 * epsarray).mean()
        )

        # Then compute MI(X;Y)
        if dim_y > 0:
            xyz_here = np.array([index for index in xyz if index == 0 or index == 1])
            array_xy = array[list(x_indices) + list(y_indices), :]
            i_xy = self.get_dependence_measure(array_xy, xyz_here)
        else:
            i_xy = 0.0

        h_x_y = h_x - i_xy

        return h_x_y

    @jit(forceobj=True)
    def get_restricted_permutation(self, T, shuffle_neighbors, neighbors, order):
        restricted_permutation = np.zeros(T, dtype=np.int32)
        used = np.array([], dtype=np.int32)

        for sample_index in order:
            m = 0
            use = neighbors[sample_index, m]

            while (use in used) and (m < shuffle_neighbors - 1):
                m += 1
                use = neighbors[sample_index, m]

            restricted_permutation[sample_index] = use
            used = np.append(used, use)

        return restricted_permutation

    def get_model_selection_criterion(self, j, parents, tau_max=0):
        """Returns a cross-validation-based score for nearest-neighbor estimates.

        Fits a nearest-neighbor model of the parents to variable j and returns
        the score. The lower, the better the fit. Here used to determine
        optimal hyperparameters in PCMCI (pc_alpha or fixed thres).

        Parameters
        ----------
        j : int
            Index of target variable in data array.
        parents : list
            List of form [(0, -1), (3, -2), ...] containing parents.
        tau_max : int, optional
            Maximum time lag. This may be used to make sure that estimates for
            different lags in X, Z, all have the same sample size. Default is 0.

        Returns
        -------
        score : float
            Model score.
        """

        # import sklearn
        from sklearn.model_selection import cross_val_score
        from sklearn.neighbors import KNeighborsRegressor

        Y = [(j, 0)]
        X = [(j, 0)]  # dummy variable here
        Z = parents
        array, xyz, _ = self.dataframe.construct_array(
            X=X,
            Y=Y,
            Z=Z,
            tau_max=tau_max,
            mask_type=self.mask_type,
            return_cleaned_xyz=False,
            do_checks=True,
            verbosity=self.verbosity,
        )
        dim, T = array.shape

        # Standardize
        array = array.astype(np.float64)
        array -= array.mean(axis=1).reshape(dim, 1)
        std = array.std(axis=1)
        for i in range(dim):
            if std[i] != 0.0:
                array[i] /= std[i]
        if np.any(std == 0.0) and self.verbosity > 0:
            warnings.warn("Possibly constant array!")
            # raise ValueError("nans after standardizing, "
            #                  "possibly constant array!")

        predictor_indices = list(np.where(xyz == 2)[0])
        predictor_array = array[predictor_indices, :].T
        # Target is only first entry of Y, ie [y]
        target_array = array[np.where(xyz == 1)[0][0], :]

        if predictor_array.size == 0:
            # Regressing on ones if empty parents
            predictor_array = np.ones(T).reshape(T, 1)

        if self.knn < 1:
            knn_here = max(1, int(self.knn * T))
        else:
            knn_here = max(1, int(self.knn))

        knn_model = KNeighborsRegressor(n_neighbors=knn_here)

        scores = cross_val_score(
            estimator=knn_model,
            X=predictor_array,
            y=target_array,
            cv=self.model_selection_folds,
            n_jobs=self.workers,
        )

        # print(scores)
        return -scores.mean()


class CMIKNN(CMIknn):
    def __init__(
        self,
        data=None,
        k=0.2,
        workers=1,
        sig_samples=500,
        cache_dir=None,
        data_hash=None,
        param_hash=None,
        early_stop=True,
        target_quantile=0.05,
        **kwargs,
    ):
        self._data = data
        self._has_nan = bool(np.isnan(data).any()) if data is not None else False
        L = 0 if data is None else data.shape[0]
        if param_hash is None:
            param_hash = f"T{L}-k{k}"
        super().__init__(
            seed=42,
            knn=k,
            workers=workers,
            sig_samples=sig_samples,
            cache_dir=cache_dir,
            data_hash=data_hash,
            param_hash=param_hash,
            mask_type=None,
            significance="shuffle_test",
            fixed_thres=None,
            sig_blocklength=3,
            confidence="bootstrap",
            conf_lev=0.9,
            conf_samples=100,
            conf_blocklength=1,
            verbosity=0,
            early_stop=early_stop,
            target_quantile=target_quantile,
            **kwargs,
        )

    def get_data(self):
        return self._data

    def set_data(self, d):
        self._data = d
        self._has_nan = bool(np.isnan(d).any()) if d is not None else False

    data = property(get_data, set_data)

    def __call__(self, X, Y, condition_set=None):
        self.n_tests += 1
        if condition_set is None:
            condition_set = []
        data = self._data
        if self._has_nan:
            cols = [X, Y] + list(condition_set)
            mask = ~np.isnan(data[:, cols]).any(axis=1)
            if mask.sum() < 30:
                return 1.0, 0.0
            data = data[mask]
        x = data[:, X]
        y = data[:, Y]
        z = data[:, condition_set]
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        if z is not None and len(z.shape) == 1:
            z = z.reshape(-1, 1)
        val, pval = super().run_test_raw(x, y, z)
        return pval, val

    def get_cmi(self, X, Y, condition_set=None):
        """
        Computes the conditional mutual information (CMI) between x and y given z.

        Parameters:
            x (ndarray): Input array for variable x.
            y (ndarray): Input array for variable y.
            z (ndarray, optional): Input array for variable z (conditioning variable).

        Returns:
            float: The computed CMI value.
        """

        if condition_set is None:
            condition_set = []
        x = self.data[:, X]
        y = self.data[:, Y]
        z = self.data[:, condition_set]
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        if z is not None and len(z.shape) == 1:
            z = z.reshape(-1, 1)
        val = self.get_dependence_raw(x, y, z)
        return val

    def get_cmi_(self, x, y, z=None):
        """
        Computes the conditional mutual information (CMI) between x and y given z.

        Parameters:
            x (ndarray): Input array for variable x.
            y (ndarray): Input array for variable y.
            z (ndarray, optional): Input array for variable z (conditioning variable).

        Returns:
            float: The computed CMI value.
        """
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        if z is not None and len(z.shape) == 1:
            z = z.reshape(-1, 1)
        val = self.get_dependence_raw(x, y, z)
        return val
