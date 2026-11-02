"""The LLM path end to end, with a fake OpenAI client (no network).

The fake 'model' is the heuristic agent rendered to text, plus one garbage
reply, so this checks prompt plumbing, text parsing and invalid-turn handling.
"""

from types import SimpleNamespace

from ledger_recon import LedgerReconEnv, run_episode
from ledger_recon.agents import HeuristicAgent
from ledger_recon.llm import LLMAgent
from ledger_recon.protocol import render


class FakeClient:
    def __init__(self, seed):
        self.inner = HeuristicAgent(seed)
        self.env_info = {}
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.last_data = None

    def create(self, model, messages, temperature, max_tokens):
        self.calls += 1
        assert messages[0]["role"] == "system" and "match(transaction_id" in messages[0]["content"]
        if self.calls == 2:
            text = "hmm let me think about this without a tool call"
        else:
            a = self.inner.act("", {"data": self.last_data})
            text = render(a["name"], a.get("args"), think="following the plan")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def test_llm_agent_text_protocol_end_to_end(monkeypatch):
    env = LedgerReconEnv("medium")
    client = FakeClient(seed=3)
    agent = LLMAgent("fake", client=client)
    # feed structured tool results back to the fake model's inner planner
    orig_step = env.step

    def step(action):
        out = orig_step(action)
        if not out[0].startswith("ERROR"):
            client.last_data = out[4]["data"]
        return out

    monkeypatch.setattr(env, "step", step)
    ro = run_episode(env, agent, seed=3)
    assert ro.breakdown["reconciliation"] == 1.0
    assert env.state.invalid_actions == 1
    assert ro.breakdown["format"] < 1.0
