"""SBP + LangChain multi-agent demo, built on SbpWorker (see sbp_worker.py).

Fully scent-driven: researcher and writer are both SbpWorker instances, neither ever
called directly. The only non-reactive step is main()'s single bootstrap emit, which
plays the role of "something outside the agent system" kicking things off — matching
the OpenAI/Hugging Face Artifactory incident this whole experiment was inspired by,
where nothing orchestrated the agents finding and building on each other's traces.
"""
import asyncio
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from sbp.agent import SbpAgent
from sbp.client import AsyncSbpClient
from sbp_worker import SbpWorker

load_dotenv()


def _model() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.environ["OPENROUTER_MODEL"],
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        timeout=60,
    )


async def main() -> None:
    researcher = SbpWorker(
        "researcher",
        _model(),
        "You are a researcher. When woken up, find a fact about Mars and call sbp_emit "
        "exactly once (trail='science.space', type='finding') to report it.",
        listens_for={"trail": "research", "signal_type": "requested", "value": 0.5},
        sbp_ops=["emit"],
    )
    writer = SbpWorker(
        "writer",
        _model(),
        "You are a writer. When woken up, write a tweet about the finding described in "
        "the blackboard state you were given, then call sbp_emit exactly once "
        "(trail='social.twitter', type='draft') with the tweet as the summary.",
        listens_for={"trail": "science.space", "signal_type": "finding", "value": 0.5},
        sbp_ops=["emit"],
    )

    done = asyncio.Event()
    observer = SbpAgent(agent_id="observer", local=True)

    @observer.when(trail="social.twitter", signal_type="draft", value=0.5)
    async def _on_tweet(trigger) -> None:
        done.set()

    worker_tasks = [asyncio.create_task(w.run()) for w in (researcher, writer)]
    observer_task = asyncio.create_task(observer.run())
    await asyncio.sleep(1)  # let scent registration land before the bootstrap emit

    bootstrap = AsyncSbpClient(local=True, agent_id="bootstrap")
    await bootstrap.connect()

    try:
        print("[bootstrap] requesting research...")
        await bootstrap.emit("research", "requested", intensity=1.0, payload={"topic": "Mars"})

        print("[system] waiting for researcher -> writer to react (stigmergy)...")
        try:
            await asyncio.wait_for(done.wait(), timeout=120)
        except asyncio.TimeoutError:
            print("[system] writer did not finish within 120s")

        result = await bootstrap.sniff(trails=["social.twitter"], types=["draft"])
        if result.pheromones:
            print(f"\ntweet: {result.pheromones[0].payload}")
        else:
            print("\nno tweet found")
    finally:
        await bootstrap.close()
        for w in (researcher, writer):
            w.stop()
        observer.stop()
        for t in worker_tasks:
            await t
        await observer_task


if __name__ == "__main__":
    asyncio.run(main())
