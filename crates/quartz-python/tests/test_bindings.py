"""pytest suite for the quartz Python bindings.

Run after `maturin develop --release -m crates/quartz-python/Cargo.toml`:
    pytest crates/quartz-python/tests -q
"""

import threading
import time

import numpy as np
import pytest

import quartz


def diag_universe():
    """3 assets, diagonal covariance [0.01, 0.04, 0.16]."""
    assets = [
        quartz.Asset("A", scores={"expected_return": 0.10}),
        quartz.Asset("B", scores={"expected_return": 0.05}),
        quartz.Asset("C", scores={"expected_return": 0.03}),
    ]
    cov = [[0.01, 0.0, 0.0], [0.0, 0.04, 0.0], [0.0, 0.0, 0.16]]
    return quartz.Universe(assets=assets, covariance=cov)


def base_restrictions(**kw):
    return quartz.Restrictions(long_only=True, fully_invested=True, **kw)


def test_min_variance_analytic():
    # Min-variance with diagonal Σ: w_i ∝ 1/σ_i² = (100, 25, 6.25) → (16, 4, 1)/21
    s = quartz.Strategy("MinVar").minimize_risk(1.0)
    sol = quartz.solve(diag_universe(), s, restrictions=base_restrictions())
    assert sol.status == quartz.SolveStatus.Optimal
    assert sol.is_optimal
    expected = [16 / 21, 4 / 21, 1 / 21]
    for w, e in zip(sol.weights_vec, expected):
        assert abs(w - e) < 1e-5
    # weights dict is ordered and keyed by id
    assert list(sol.weights.keys()) == ["A", "B", "C"]
    assert sol.solve_time_s > 0
    assert sol.iterations > 0


def test_numpy_and_list_input_parity():
    cov_list = [[0.01, 0.0, 0.0], [0.0, 0.04, 0.0], [0.0, 0.0, 0.16]]
    assets = lambda: [  # noqa: E731
        quartz.Asset(t, scores={"expected_return": r})
        for t, r in [("A", 0.10), ("B", 0.05), ("C", 0.03)]
    ]
    u_list = quartz.Universe(assets=assets(), covariance=cov_list)
    u_np = quartz.Universe(assets=assets(), covariance=np.array(cov_list))
    # int numpy arrays go through the Rows fallback
    u_int = quartz.Universe(assets=assets(), covariance=np.eye(3, dtype=int))

    s = quartz.Strategy("MinVar").minimize_risk(1.0)
    w_list = quartz.solve(u_list, s, restrictions=base_restrictions()).weights_vec
    w_np = quartz.solve(u_np, s, restrictions=base_restrictions()).weights_vec
    assert w_list == w_np
    assert u_int.n_assets == 3


def test_factor_model_matches_densified():
    B = [[1.0, 0.2], [0.8, -0.1], [0.5, 0.7]]
    F = [[0.04, 0.01], [0.01, 0.02]]
    D = [0.01, 0.02, 0.015]
    n, k = 3, 2
    sigma = [
        [
            sum(B[i][r] * F[r][c] * B[j][c] for r in range(k) for c in range(k))
            + (D[i] if i == j else 0.0)
            for j in range(n)
        ]
        for i in range(n)
    ]
    assets = lambda: [  # noqa: E731
        quartz.Asset(t, scores={"expected_return": r})
        for t, r in [("A", 0.10), ("B", 0.05), ("C", 0.03)]
    ]
    u_factor = quartz.Universe(assets=assets(), factor_model=(B, F, D))
    u_full = quartz.Universe(assets=assets(), covariance=sigma)

    s = quartz.Strategy("Mix").minimize_risk(0.7).maximize("expected_return", 0.3)
    sol_f = quartz.solve(u_factor, s, restrictions=base_restrictions())
    sol_d = quartz.solve(u_full, s, restrictions=base_restrictions())
    assert sol_f.status == quartz.SolveStatus.Optimal
    for wf, wd in zip(sol_f.weights_vec, sol_d.weights_vec):
        assert abs(wf - wd) < 1e-6
    assert abs(sol_f.portfolio_scores["financial_risk"] - sol_d.portfolio_scores["financial_risk"]) < 1e-8


def test_restrictions():
    s = quartz.Strategy("MinVar").minimize_risk(1.0)
    r = base_restrictions(max_single_weight=0.5, exclude_assets=["A"])
    sol = quartz.solve(diag_universe(), s, restrictions=r)
    assert sol.status == quartz.SolveStatus.Optimal
    assert abs(sol.weights["A"]) < 1e-7
    assert max(sol.weights_vec) <= 0.5 + 1e-6


def test_group_and_score_constraints():
    assets = [
        quartz.Asset("A", tags={"currency": "USD"}, scores={"expected_return": 0.10, "esg": 3.0}),
        quartz.Asset("B", tags={"currency": "EUR"}, scores={"expected_return": 0.05, "esg": 8.0}),
        quartz.Asset("C", tags={"currency": "EUR"}, scores={"expected_return": 0.03, "esg": 9.0}),
    ]
    cov = np.diag([0.04, 0.02, 0.01])
    u = quartz.Universe(assets=assets, covariance=cov)
    s = (
        quartz.Strategy("ESG")
        .minimize_risk(0.5)
        .maximize("expected_return", 0.5)
        .group("currency", "EUR", 0.5, 1.0)
        .score_min("esg", 6.0)
    )
    sol = quartz.solve(u, s, restrictions=base_restrictions())
    assert sol.status == quartz.SolveStatus.Optimal
    eur = sol.weights["B"] + sol.weights["C"]
    assert eur >= 0.5 - 1e-6
    assert sol.portfolio_scores["esg"] >= 6.0 - 1e-6


def test_infeasible_is_status_not_exception():
    s = quartz.Strategy("Impossible").minimize_risk(0.5).maximize("expected_return", 0.5) \
        .score_min("expected_return", 99.0)
    sol = quartz.solve(diag_universe(), s, restrictions=base_restrictions())
    assert sol.status == quartz.SolveStatus.Infeasible
    assert not sol.is_optimal


def test_error_mapping():
    assert issubclass(quartz.QuartzError, ValueError)
    assets = [quartz.Asset("A"), quartz.Asset("B"), quartz.Asset("C")]
    with pytest.raises(quartz.QuartzError) as exc:
        quartz.Universe(assets=assets, covariance=[[0.04, 0.0], [0.0, 0.01]])
    assert "2" in str(exc.value) and "3" in str(exc.value)
    with pytest.raises(quartz.QuartzError):
        quartz.Universe(assets=assets)  # neither covariance nor factor model


def favoring_universe(hot):
    assets = [
        quartz.Asset(f"A{i}", scores={"expected_return": 0.50 if i == hot else 0.01})
        for i in range(3)
    ]
    return quartz.Universe(assets=assets, covariance=np.diag([0.04] * 3))


def test_batch_order_and_error_isolation():
    s = quartz.Strategy("Batch").minimize_risk(0.3).maximize("expected_return", 0.7)
    problems = []
    for i in range(8):
        u = favoring_universe(i % 3)
        if i == 4:
            # wrong-length turnover -> deterministic per-item error
            problems.append(quartz.Problem(u, s, turnover=([0.5, 0.5], 0.1)))
        else:
            problems.append((u, s))

    results = quartz.solve_batch(problems, restrictions=base_restrictions())
    assert len(results) == 8
    for i, r in enumerate(results):
        if i == 4:
            assert isinstance(r, quartz.QuartzError)
        else:
            assert r.status == quartz.SolveStatus.Optimal
            weights = r.weights_vec
            assert weights.index(max(weights)) == i % 3  # input order preserved


def test_batch_infeasible_is_solution():
    s = quartz.Strategy("Impossible").minimize_risk(1.0).score_min("expected_return", 99.0)
    results = quartz.solve_batch([(diag_universe(), s)], restrictions=base_restrictions())
    assert results[0].status == quartz.SolveStatus.Infeasible


def test_batch_releases_gil():
    # A background Python thread must keep running while the batch solves.
    s = quartz.Strategy("Batch").minimize_risk(0.3).maximize("expected_return", 0.7)
    problems = [(favoring_universe(i % 3), s) for i in range(64)]

    ticks = []
    stop = threading.Event()

    def ticker():
        while not stop.is_set():
            ticks.append(1)
            time.sleep(0.0005)

    t = threading.Thread(target=ticker)
    t.start()
    try:
        results = quartz.solve_batch(problems, restrictions=base_restrictions())
    finally:
        stop.set()
        t.join()
    assert all(r.status == quartz.SolveStatus.Optimal for r in results)
    assert len(ticks) > 0  # GIL was released during solving


def test_sweep():
    s = quartz.Strategy("Base").minimize_risk(0.5).maximize("expected_return", 0.5)
    fr = quartz.sweep(
        diag_universe(), s, "expected_return", "financial_risk",
        n_points=5, restrictions=base_restrictions(),
    )
    assert fr.objective_dims == ["expected_return", "financial_risk"]
    assert len(fr) + fr.n_skipped == 5
    for p in fr.points:
        assert abs(sum(p.dimension_weights.values()) - 1.0) < 1e-9
    assert any(p.is_efficient for p in fr.points)
    # returns increase along the sweep (alpha on expected_return grows)
    rets = [p.portfolio_scores["expected_return"] for p in fr.points]
    assert rets[-1] >= rets[0] - 1e-8

    with pytest.raises(quartz.QuartzError):
        quartz.sweep(diag_universe(), s, "expected_return", "financial_risk",
                     n_points=1, restrictions=base_restrictions())


def test_simplex_grid():
    assets = [
        quartz.Asset("A", scores={"expected_return": 0.10, "esg": 3.0}),
        quartz.Asset("B", scores={"expected_return": 0.05, "esg": 8.0}),
    ]
    u = quartz.Universe(assets=assets, covariance=np.diag([0.04, 0.01]))
    s = (quartz.Strategy("3dim").minimize_risk(0.4)
         .maximize("expected_return", 0.3).maximize("esg", 0.3))
    fr = quartz.simplex_grid(u, s, resolution=4, restrictions=base_restrictions())
    # C(4+3-1, 3-1) = 15 compositions
    assert len(fr) + fr.n_skipped == 15


def test_tactic():
    assets = [
        quartz.Asset("A", tags={"currency": "USD"}, scores={"expected_return": 0.10}),
        quartz.Asset("B", tags={"currency": "EUR"}, scores={"expected_return": 0.05}),
    ]
    u = quartz.Universe(assets=assets, covariance=np.diag([0.04, 0.01]))
    s = (quartz.Strategy("Base").minimize_risk(0.5).maximize("expected_return", 0.5)
         .group("currency", "EUR", 0.0, 1.0))
    t = quartz.Tactic("Q3").override_group("currency", "EUR", 0.6, 1.0)
    sol = quartz.solve(u, s, tactic=t, restrictions=base_restrictions())
    assert sol.status == quartz.SolveStatus.Optimal
    assert sol.weights["B"] >= 0.6 - 1e-6
