import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from talker.errands import ErrandRunner
from talker.orchestrator import Orchestrator, Step, parse_plan

ASKED = "I couldn't finish: the agent needed an answer or a confirmation and nobody was there to give it."
PROPOSAL = "About to call 'send_sms':\n  to_number: +12125550199\n  message: Hilton Head 74-80F, showers.\n\nProceed? (yes/no)"


class Agent:
    """A relay stand-in: every POST is a task the test then finishes, fails or makes ask."""
    def __init__(self):
        self.tasks, self.order, self.down = {}, [], False

    def fetch(self, url, method="GET", body=None, timeout=5.0):
        if self.down:
            raise RuntimeError(f"{url} not reachable: refused")
        if method == "POST":
            tid = f"t{len(self.order) + 1}"
            self.tasks[tid] = {"state": "running", "summary": "", "result": "", "task": body["task"]}
            self.order.append(tid)
            return {"id": tid}
        return self.tasks[url.rsplit("/", 1)[1]]

    def last(self):
        return self.tasks[self.order[-1]]

    def end(self, tid=None, state="done", summary="", result=""):
        self.tasks[tid or self.order[-1]].update(state=state, summary=summary, result=result)


def rig(steps, mode="auto"):
    agent, said = Agent(), []
    runner = ErrandRunner("http://agent:8080", fetch=agent.fetch)
    orch = Orchestrator(runner, planner=lambda goal, ctx: [Step(**s) for s in steps],
                        announce=said.append, mode=mode)
    runner.on_done, runner.on_fail = orch.on_errand_done, orch.on_errand_failed
    return agent, runner, orch, said


def spin(runner, n=3):
    for _ in range(n):
        runner.cycle()


def test_steps_run_in_order_and_pass_results_forward():
    agent, runner, orch, said = rig([dict(n=1, do="get the Hilton Head forecast"),
                                     dict(n=2, do="text the forecast to +12125550199", after=[1],
                                          kind="action", approved=True)])
    g = orch.start_goal("weather to my phone", plan_now=True)
    spin(runner)
    assert len(agent.order) == 1                              # step 2 waits for step 1
    agent.end(summary="74 to 80, showers likely.", result="Hilton Head: 74-80F, 48% rain.")
    spin(runner)
    assert len(agent.order) == 2
    second = agent.last()["task"]
    assert "Hilton Head: 74-80F" in second and "APPROVED IN ADVANCE" in second
    agent.end(summary="Text sent.")
    spin(runner)
    assert g.state == "done" and len(said) == 1               # one announcement for the whole goal
    assert "74 to 80" in said[0] and "Text sent" in said[0]


def test_an_approved_action_is_resent_when_the_agent_asks_then_reported_if_it_insists():
    agent, runner, orch, said = rig([dict(n=1, do="get forecast"),
                                     dict(n=2, do="text it", after=[1], kind="action", approved=True)])
    g = orch.start_goal("weather to my phone", plan_now=True)
    spin(runner); agent.end(summary="74 to 80, showers.")
    spin(runner); agent.end(state="failed", summary=ASKED, result=PROPOSAL)
    spin(runner)
    assert len(agent.order) == 3 and "+12125550199" in agent.last()["task"]   # resent with its own proposal
    assert said == []                                         # nobody was bothered
    agent.end(state="failed", summary=ASKED, result=PROPOSAL)  # it asks again: it confirms at its own end
    spin(runner)
    assert g.state == "failed" and len(said) == 1
    assert "74 to 80" in said[0] and "own end" in said[0]     # the useful part is not lost


def test_an_action_the_request_did_not_cover_asks_the_owner_once():
    agent, runner, orch, said = rig([dict(n=1, do="email the supplier", kind="action", approved=False)])
    g = orch.start_goal("deal with the supplier", plan_now=True)
    spin(runner); agent.end(state="failed", summary=ASKED, result="About to send email to bob@x.com. Proceed? (yes/no)")
    spin(runner)
    assert g.state == "needs_input" and len(said) == 1 and f"{{{{approve {g.id}}}}}" in said[0]
    assert orch.decide(g.id, True).endswith("approved")
    spin(runner)
    assert "APPROVED IN ADVANCE" in agent.last()["task"] and "bob@x.com" in agent.last()["task"]
    agent.end(summary="Email sent to Bob.")
    spin(runner)
    assert g.state == "done" and "Email sent" in said[-1]


def test_declining_skips_the_step_and_what_depends_on_it():
    agent, runner, orch, said = rig([dict(n=1, do="book it", kind="action"),
                                     dict(n=2, do="tell Bob", after=[1], kind="action")])
    g = orch.start_goal("book and tell", plan_now=True)
    spin(runner); agent.end(state="failed", summary=ASKED, result="Book for $300? (yes/no)")
    spin(runner)
    orch.decide(g.id, False)
    assert [s.state for s in g.steps] == ["skipped", "skipped"] and g.state == "done"
    assert "Not done" in said[-1]


def test_transient_failures_are_retried_quietly_and_real_ones_get_one_more_try():
    agent, runner, orch, said = rig([dict(n=1, do="look it up")])
    g = orch.start_goal("look it up", plan_now=True)
    spin(runner); agent.end(state="failed", summary="The relay restarted before this finished.")
    spin(runner)
    assert len(agent.order) == 2 and said == []
    agent.end(state="failed", summary="The task failed: page not found")
    spin(runner)
    assert len(agent.order) == 3 and "A previous attempt failed" in agent.last()["task"]
    agent.end(state="failed", summary="The task failed: page not found")
    spin(runner)
    assert g.state == "failed" and len(said) == 1 and "page not found" in said[0]


def test_ask_mode_never_counts_the_request_as_approval():
    agent, runner, orch, said = rig([dict(n=1, do="text it", kind="action", approved=True)], mode="ask")
    g = orch.start_goal("text me", plan_now=True)
    spin(runner)
    assert "APPROVED IN ADVANCE" not in agent.last()["task"]
    agent.end(state="failed", summary=ASKED, result=PROPOSAL)
    spin(runner)
    assert g.state == "needs_input"


class AgentV2(Agent):
    """Protocol 2: approvals echoed, needs_input with question/action, /answer resumes, error_kind."""
    def __init__(self):
        super().__init__()
        self.answers, self.posts = [], []

    def fetch(self, url, method="GET", body=None, timeout=5.0):
        if url.endswith("/health"):
            return {"ok": True, "protocol": 2, "features": ["approvals", "needs_input", "answer", "error_kind"]}
        if method == "POST" and url.endswith("/answer"):
            tid = url.rsplit("/", 2)[1]
            if self.tasks[tid]["state"] != "needs_input":
                raise RuntimeError(f"409 from {url}: task is not waiting for input")
            self.answers.append((tid, body["answer"], body["by"]))
            self.tasks[tid].update(state="running", question="", action=None)
            return {"id": tid, "state": "running"}
        if method == "POST":
            self.posts.append(body)
        return super().fetch(url, method, body, timeout)

    def ask(self, question, action=None, options=("yes", "no")):
        self.last().update(state="needs_input", question=question, action=action, options=list(options),
                           summary="I need approval before I send sms.", result="Hilton Head 74-80F, showers.")


def rig2(steps, mode="auto"):
    agent, said = AgentV2(), []
    runner = ErrandRunner("http://agent:8080", fetch=agent.fetch)
    orch = Orchestrator(runner, planner=lambda goal, ctx: [Step(**s) for s in steps], announce=said.append, mode=mode)
    runner.on_done, runner.on_fail, runner.on_needs_input = orch.on_errand_done, orch.on_errand_failed, orch.on_errand_needs_input
    return agent, runner, orch, said


SMS = {"tool": "send_sms", "args": {"to_number": "+12125550199", "message": "Hilton Head 74-80F"}}


def test_v2_approved_action_goes_out_with_approvals_and_a_covered_question_is_answered_by_clara():
    agent, runner, orch, said = rig2([dict(n=1, do="get forecast"),
                                      dict(n=2, do="text the forecast to 212-555-0199", after=[1], kind="action", approved=True)])
    g = orch.start_goal("weather in Hilton Head, texted to me at 212-555-0199", plan_now=True)
    spin(runner); agent.end(summary="74 to 80, showers.")
    spin(runner)
    assert agent.posts[-1]["approvals"] == [{"tool": "send_sms", "args": {"to_number": "+12125550199"}}]
    agent.ask("Send this text to +12125550199?", SMS)       # an agent that asks anyway
    spin(runner)
    assert agent.answers == [(agent.order[-1], "yes", "clara")] and said == []
    agent.end(summary="Text sent.")
    spin(runner)
    assert g.state == "done" and len(agent.order) == 2       # resumed, never resubmitted
    assert "Text sent" in said[0]


def test_v2_a_different_number_is_never_answered_for_the_owner():
    agent, runner, orch, said = rig2([dict(n=1, do="text the forecast to 212-555-0199", kind="action", approved=True)])
    g = orch.start_goal("text me at 212-555-0199", plan_now=True)
    spin(runner)
    agent.ask("Send this text to +12125550000?", {"tool": "send_sms", "args": {"to_number": "+12125550000"}})
    spin(runner)
    assert agent.answers == [] and g.state == "needs_input" and "+12125550000" in said[0]
    orch.decide(g.id, False)
    spin(runner)
    assert agent.answers == [(agent.order[-1], "no", "owner")]
    agent.end(summary="Forecast found; the text was not sent (declined).")
    spin(runner)
    assert g.state == "done" and "not sent" in said[-1]


def test_v2_error_kinds_drive_retries_and_reports():
    agent, runner, orch, said = rig2([dict(n=1, do="look it up")])
    g = orch.start_goal("look it up", plan_now=True)
    spin(runner); agent.end(state="failed", summary="backend hiccup", result="")
    agent.last()["error_kind"] = "transient"
    spin(runner)
    assert len(agent.order) == 2 and said == []
    agent.end(state="failed", summary="Gave up waiting.", result="partial: 74-80F")
    agent.last().update(error_kind="needs_input_expired", question="Which Hilton Head hotel?")
    spin(runner)
    assert g.state == "failed" and len(agent.order) == 2      # expired is reported, not retried
    assert "nobody answered" in said[0] and "Which Hilton Head hotel?" in said[0]


def test_parse_plan_falls_back_to_one_step_and_cleans_dependencies():
    assert [s.do for s in parse_plan("not json", "the whole goal")] == ["the whole goal"]
    steps = parse_plan('{"steps": [{"n": 1, "do": "find it"}, {"n": 2, "do": "send it", "after": [1, 7, 2],'
                       ' "kind": "action", "approved": true}, {"n": 3, "do": "", "after": []}]}', "g")
    assert [s.n for s in steps] == [1, 2] and steps[1].after == [1] and steps[1].approved
    info_approved = parse_plan('{"steps": [{"n": 1, "do": "read", "kind": "info", "approved": true}]}', "g")
    assert info_approved[0].approved is False                 # only actions carry approval


def test_a_restart_resumes_open_goals_instead_of_losing_them(tmp_path):
    # Six goals were lost to restarts, flight searches among them, while the agent kept working.
    path = str(tmp_path / "goals.json")
    agent, said = Agent(), []
    steps = [dict(n=1, do="find nonstop flights to Las Vegas Oct 20"),
             dict(n=2, do="text the best one to +12125550199", after=[1], kind="action", approved=True)]

    def build():
        runner = ErrandRunner("http://agent:8080", fetch=agent.fetch)
        plan = lambda goal, ctx: ([Step(n=1, do=goal)] if goal == "already finished"
                                  else [Step(**s) for s in steps])
        orch = Orchestrator(runner, planner=plan, announce=said.append, state_path=path)
        runner.on_done, runner.on_fail = orch.on_errand_done, orch.on_errand_failed
        return runner, orch

    runner, orch = build()
    g = orch.start_goal("flights to Vegas, text me the best", plan_now=True)
    spin(runner)
    assert agent.order == ["t1"]
    finished = orch.start_goal("already finished", plan_now=True)
    spin(runner)
    agent.end("t2", summary="done earlier")
    spin(runner)
    del runner, orch                                          # the service restarts

    agent.end("t1", summary="Southwest 1234, 9:05 am, $129.")  # the agent finished meanwhile
    runner, orch = build()
    resumed = orch.restore()
    assert [r.id for r in resumed] == [g.id]                  # the finished goal is not resumed
    spin(runner)
    assert len(agent.order) == 3                              # step 2 went out, with step 1's result
    assert "Southwest 1234" in agent.last()["task"]
    agent.end(summary="Texted.")
    spin(runner)
    assert orch.goals[g.id].state == "done" and any("finished" in s and g.id in s for s in said)


def test_a_step_that_never_reached_the_agent_is_sent_again(tmp_path):
    path = str(tmp_path / "goals.json")
    agent = Agent()
    agent.down = True
    runner = ErrandRunner("http://agent:8080", fetch=agent.fetch)
    orch = Orchestrator(runner, planner=lambda goal, ctx: [Step(n=1, do="look it up")],
                        announce=lambda t: None, state_path=path)
    orch.start_goal("look it up", plan_now=True)
    spin(runner)                                              # the agent is down: never posted
    assert agent.order == []
    agent.down = False
    runner2 = ErrandRunner("http://agent:8080", fetch=agent.fetch)
    orch2 = Orchestrator(runner2, planner=lambda goal, ctx: [], announce=lambda t: None, state_path=path)
    assert len(orch2.restore()) == 1
    spin(runner2)
    assert agent.order == ["t1"] and agent.last()["task"].startswith("look it up")
