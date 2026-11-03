"""Tests that pin down the properties a training run depends on."""

import statistics

import pytest

from ledger_recon import LedgerReconEnv, NO_MATCH, generate, run_episode
from ledger_recon.agents import AGENTS, AmountOnlyAgent, FlagAllAgent, HeuristicAgent, RandomAgent, TrustRefAgent
from ledger_recon.protocol import parse, render

SEEDS = range(20)


def mean_reward(agent_cls, difficulty, seeds=SEEDS):
    return statistics.mean(run_episode(LedgerReconEnv(difficulty), agent_cls(s), s).reward for s in seeds)


# --- determinism: GRPO groups must see the same world -------------------------

def test_same_seed_same_scenario():
    a, b = generate(3, "hard"), generate(3, "hard")
    assert a.invoices == b.invoices and a.transactions == b.transactions and a.gold == b.gold


def test_different_seeds_differ():
    assert generate(1, "medium").transactions != generate(2, "medium").transactions


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_gold_references_real_invoices(difficulty):
    for s in SEEDS:
        sc = generate(s, difficulty)
        ids = {i.invoice_id for i in sc.invoices}
        for label in sc.gold.values():
            assert label == frozenset({NO_MATCH}) or label <= ids


def test_difficulty_scales_size():
    sizes = [len(generate(0, d).transactions) for d in ("easy", "medium", "hard")]
    assert sizes == sorted(sizes) and sizes[0] < sizes[-1]


# --- reward hacking probes -------------------------------------------------------

def test_submit_immediately_scores_zero_task_reward():
    env = LedgerReconEnv("medium")
    env.reset(seed=0)
    _, reward, done, _, info = env.step({"name": "submit"})
    assert done and info["reward_breakdown"]["reconciliation"] == 0.0


@pytest.mark.parametrize("difficulty", ["medium", "hard"])
def test_cheap_policies_score_low(difficulty):
    assert mean_reward(RandomAgent, difficulty) < 0.35
    assert mean_reward(FlagAllAgent, difficulty) < 0.35


@pytest.mark.parametrize("naive", [AmountOnlyAgent, TrustRefAgent])
def test_naive_solvers_fail_on_hard(naive):
    assert mean_reward(naive, "hard") < mean_reward(HeuristicAgent, "hard") - 0.2


def test_matching_everything_to_one_txn_is_punished():
    env = LedgerReconEnv("easy")
    env.reset(seed=1)
    sc = env.state.scenario
    t = sc.transactions[0]
    env.step({"name": "match", "args": {"transaction_id": t.transaction_id,
                                         "invoice_ids": [i.invoice_id for i in sc.invoices]}})
    br = env.breakdown()
    assert br.reconciliation < 1 / len(sc.transactions) * 0.5


def test_cannot_spray_one_invoice_across_many_payments():
    env = LedgerReconEnv("easy")
    env.reset(seed=2)
    sc = env.state.scenario
    paid = [t for t in sc.transactions if sc.kinds[t.transaction_id] == "exact"]
    inv = next(iter(sc.gold[paid[0].transaction_id]))
    env.step({"name": "match", "args": {"transaction_id": paid[0].transaction_id, "invoice_ids": [inv]}})
    obs, *_ = env.step({"name": "match", "args": {"transaction_id": paid[1].transaction_id, "invoice_ids": [inv]}})
    assert "already fully paid" in obs


# --- solvability -----------------------------------------------------------------

@pytest.mark.parametrize("difficulty,floor", [("easy", 1.05), ("medium", 1.0), ("hard", 1.0)])
def test_heuristic_solves_through_tools(difficulty, floor):
    assert mean_reward(HeuristicAgent, difficulty) > floor


def test_perfect_answer_gets_max_reward():
    env = LedgerReconEnv("hard")
    env.reset(seed=5)
    sc = env.state.scenario
    for tid, label in sc.gold.items():
        if label == frozenset({NO_MATCH}):
            env.step({"name": "flag", "args": {"transaction_id": tid, "reason": "noise"}})
        else:
            env.step({"name": "match", "args": {"transaction_id": tid, "invoice_ids": sorted(label)}})
    _, reward, *_ , info = env.step({"name": "submit"})
    assert info["reward_breakdown"]["solved"] and reward == pytest.approx(1.1)


# --- action protocol / invalid actions --------------------------------------------

def test_invalid_text_is_penalised_not_fatal():
    env = LedgerReconEnv("easy")
    env.reset(seed=0)
    obs, _, done, _, _ = env.step("I think I'll match the first one")
    assert obs.startswith("ERROR") and not done
    obs, _, done, _, _ = env.step(render("list_invoices", {"page": 1}))
    assert "invoices" in obs and not done
    assert env.state.invalid_actions == 1


def test_unknown_tool_and_bad_args():
    env = LedgerReconEnv("easy")
    env.reset(seed=0)
    assert env.step({"name": "delete_ledger"})[0].startswith("ERROR")
    assert env.step({"name": "match", "args": {"txn": "x"}})[0].startswith("ERROR")


def test_parse_roundtrip_and_errors():
    p = parse(render("flag", {"transaction_id": "TXN-1", "reason": "fee"}, think="it's a fee"))
    assert p.ok and p.name == "flag" and p.args["transaction_id"] == "TXN-1"
    assert not parse("<tool>{bad json}</tool>").ok
    assert not parse('<tool>{"name":"a"}</tool><tool>{"name":"b"}</tool>').ok


def test_turn_budget_truncates():
    env = LedgerReconEnv("easy", max_turns=3)
    env.reset(seed=0)
    for _ in range(2):
        assert not env.step({"name": "list_invoices"})[2]
    _, _, terminated, truncated, info = env.step({"name": "list_invoices"})
    assert truncated and not terminated and "reward_breakdown" in info


def test_unmatch_restores_state():
    env = LedgerReconEnv("easy")
    env.reset(seed=0)
    t = env.state.scenario.transactions[0].transaction_id
    env.step({"name": "flag", "args": {"transaction_id": t, "reason": "x"}})
    env.step({"name": "unmatch", "args": {"transaction_id": t}})
    assert t not in env.state.flags and t not in env.state.matches


def test_all_agents_terminate():
    for name, cls in AGENTS.items():
        for d in ("easy", "hard"):
            r = run_episode(LedgerReconEnv(d), cls(0), 0, name)
            assert r.breakdown


def test_garbage_arguments_never_crash_the_env():
    """Found by a real 1.5B model sending amount="". Every tool, junk values."""
    import itertools
    junk = ["", "two", None, -1, 3.5, [], {}, ["x"], {"a": 1}, "TXN-00000", True]
    env = LedgerReconEnv("hard")
    obs, info = env.reset(seed=3)
    tools = ["list_transactions", "list_invoices", "search_invoices", "get_fx_rate",
             "match", "unmatch", "flag"]
    keys = ["status", "page", "customer", "amount", "currency", "from_currency", "to_currency",
            "transaction_id", "invoice_ids", "reason"]
    for tool, key, val in itertools.product(tools, keys, junk):
        if env.state.done:
            env.reset(seed=3)
        obs, *_ = env.step({"name": tool, "args": {key: val}})
        assert isinstance(obs, str)
    env.reset(seed=3)
    obs, *_ = env.step({"name": "search_invoices", "args": {"amount": "", "currency": "PLN"}})
    assert obs.startswith("ERROR")
