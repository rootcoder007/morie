from morie.fn.electoral import gelman_king_incumbency, network_polarization, proportional_seats, runoff_winner


def test_dhondt_and_sainte_lague_textbook():
    assert proportional_seats([100000, 80000, 30000, 20000], 8).seats == [4, 3, 1, 0]
    assert proportional_seats([100000, 80000, 30000, 20000], 8, "sainte_lague").seats == [3, 3, 1, 1]
    # Hare versus Droop on the classic 6-seat example
    v = [47000, 16000, 15800, 12000, 6100, 3100]
    assert proportional_seats(v, 10, "hare").seats == [5, 2, 1, 1, 1, 0]
    assert proportional_seats(v, 10, "droop").seats == [5, 2, 2, 1, 0, 0]
    assert proportional_seats(v, 10, "dhondt", threshold=0.05).seats[5] == 0


def test_runoff():
    b = [[0, 1, 2]] * 4 + [[1, 2, 0]] * 3 + [[2, 1, 0]] * 2
    r = runoff_winner(b)
    assert r.first_round == [4, 3, 2] and r.second_round == {0: 4, 1: 5} and r.winner == 1
    assert runoff_winner([[2, 0]] * 5 + [[0]] * 2).winner == 2


def test_gelman_king_normal_equations():
    v = [0.55, 0.62, 0.41, 0.48, 0.66, 0.37, 0.52, 0.45]
    vl = [0.52, 0.58, 0.45, 0.50, 0.61, 0.40, 0.49, 0.47]
    P = [1, 1, -1, -1, 1, -1, 1, -1]
    ident = [1, 1, -1, 0, 1, -1, 0, -1]
    r = gelman_king_incumbency(v, vl, P, ident)
    X = [[1, vl[i], P[i], ident[i]] for i in range(8)]
    res = [v[i] - sum(X[i][a] * r.coefficients[a] for a in range(4)) for i in range(8)]
    for a in range(4):
        assert abs(sum(X[i][a] * res[i] for i in range(8))) < 1e-12


def test_network_polarization_modularity():
    V = [[1, 1, 0, 1, None], [1, 1, 0, 0, 1], [0, 0, 1, 1, 0], [0, 1, 1, None, 0], [1, 0, 0, 1, 1]]
    lab = ["D", "D", "R", "R", "D"]
    r = network_polarization(V, lab)
    A = r.agreement
    k = [sum(row) for row in A]
    m2 = sum(k)
    q = sum(A[i][j] - k[i] * k[j] / m2 for i in range(5) for j in range(5) if lab[i] == lab[j]) / m2
    assert abs(r.modularity - q) < 1e-12
    assert abs(A[0][3] - 1 / 3) < 1e-15
