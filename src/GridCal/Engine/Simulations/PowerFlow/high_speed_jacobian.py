# Copyright 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for for Energy Economics
# and Energy System Technology (IEE) Kassel and individual contributors (see AUTHORS file for details).
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted
# provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions
# and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of
# conditions and the following disclaimer in the documentation and/or other materials provided
# with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to
# endorse or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import numba as nb
import numpy as np
import scipy.sparse as sp

@nb.njit()
def jacobian_numba(nbus, Gi, Gp, Gx, Bx, P, Q, E, F, pq, pvpq):
    """
    Compute the Tinney version of the AC jacobian without any sin, cos or abs
    (Lynn book page 89)
    :param G: Conductance matrix in CSC format
    :param B: Susceptance matrix in CSC format
    :param P: Real computed power
    :param Q: Imaginary computed power
    :param E: Real voltage
    :param F: Imaginary voltage
    :param pq: array pf pq indices
    :param pv: array of pv indices
    :return: CSC Jacobian matrix
    """
    npqpv = len(pvpq)
    n_rows = len(pvpq) + len(pq)
    n_cols = len(pvpq) + len(pq)

    nnz = 0
    p = 0
    Jx = np.empty(len(Gx) * 4, dtype=nb.float64)  # data
    Ji = np.empty(len(Gx) * 4, dtype=nb.int32)  # indices
    Jp = np.empty(n_cols + 1, dtype=nb.int32)  # pointers
    Jp[p] = 0

    # generate lookup for the non immediate axis (for CSC it is the rows) -> index lookup
    lookup_pqpv = np.zeros(nbus, dtype=nb.int32)
    lookup_pqpv[pvpq] = np.arange(len(pvpq), dtype=nb.int32)

    lookup_pq = np.zeros(nbus, dtype=nb.int32)
    lookup_pq[pq] = np.arange(len(pq), dtype=nb.int32)

    for j in pvpq:  # sliced columns

        # fill in J1
        for k in range(Gp[j], Gp[j + 1]):  # rows of A[:, j]

            # row index translation to the "rows" space
            i = Gi[k]
            ii = lookup_pqpv[i]

            if pvpq[ii] == i:  # rows
                # entry found
                if i != j:
                    Jx[nnz] = F[i] * (Gx[k] * E[j] - Bx[k] * F[j]) - \
                              E[i] * (Bx[k] * E[j] + Gx[k] * F[j])
                else:
                    Jx[nnz] = -Q[i] - Bx[k] * (E[i] + F[i])

                Ji[nnz] = ii
                nnz += 1

        # fill in J3
        for k in range(Gp[j], Gp[j + 1]):  # rows of A[:, j]

            # row index translation to the "rows" space
            i = Gi[k]
            ii = lookup_pq[i]

            if pq[ii] == i:  # rows
                # entry found
                if i != j:
                    Jx[nnz] = - E[i] * (Gx[k] * E[j] - Bx[k] * F[j]) \
                              - F[i] * (Bx[k] * E[j] + Gx[k] * F[j])
                else:
                    Jx[nnz] = P[i] - Gx[k] * (E[i] + F[i])

                Ji[nnz] = ii + npqpv
                nnz += 1

        p += 1
        Jp[p] = nnz

    # J2 and J4
    for j in pq:  # sliced columns

        # fill in J2
        for k in range(Gp[j], Gp[j + 1]):  # rows of A[:, j]

            # row index translation to the "rows" space
            i = Gi[k]
            ii = lookup_pqpv[i]

            if pvpq[ii] == i:  # rows
                # entry found
                if i != j:
                    Jx[nnz] = E[i] * (Gx[k] * E[j] - Bx[k] * F[j]) \
                              + F[i] * (Bx[k] * E[j] + Gx[k] * F[j])
                else:
                    Jx[nnz] = P[i] + Gx[k] * (E[i] + F[i])

                Ji[nnz] = ii
                nnz += 1

        # fill in J4
        for k in range(Gp[j], Gp[j + 1]):  # rows of A[:, j]

            # row index translation to the "rows" space
            i = Gi[k]
            ii = lookup_pq[i]

            if pq[ii] == i:  # rows
                # entry found
                if i != j:
                    Jx[nnz] = F[i] * (Gx[k] * E[j] - Bx[k] * F[j]) \
                              - E[i] * (Bx[k] * E[j] + Gx[k] * F[j])
                else:
                    Jx[nnz] = Q[i] - Bx[k] * (E[i] + F[i])

                Ji[nnz] = ii + npqpv
                nnz += 1

        p += 1
        Jp[p] = nnz

    # last pointer entry
    Jp[p] = nnz

    # reseize
    # Jx = np.resize(Jx, nnz)
    # Ji = np.resize(Ji, nnz)

    return Jx, Ji, Jp, n_rows, n_cols, nnz


def AC_jacobian(Y, S, V, pq, pvpq):

    Jx, Ji, Jp, n_rows, n_cols, nnz = jacobian_numba(nbus=len(S),
                                                     Gi=Y.indices, Gp=Y.indptr, Gx=Y.data.real,
                                                     Bx=Y.data.imag, P=S.real, Q=S.imag,
                                                     E=V.real, F=V.imag,
                                                     pq=pq, pvpq=pvpq)
    Jx = np.resize(Jx, nnz)
    Ji = np.resize(Ji, nnz)

    return sp.csc_matrix((Jx, Ji, Jp), shape=(n_rows, n_cols))
