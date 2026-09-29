"""Copy this project and implement RuntimeAdapter 1.0 for your runtime."""
from almm_adapter.contract import (ModelRequest, Probe, ProbeResult, RunManifest,
                                   Turn, TurnResult)


class Adapter:
    def initialize(self, runManifest: RunManifest) -> None:
        raise NotImplementedError('Implement initialize(runManifest): validate contractVersion and reset ALL run state')

    def handleTurn(self, turn: Turn) -> TurnResult:
        raise NotImplementedError('Implement handleTurn(turn): call model proxy and return response plus tiered requests')

    def answerProbe(self, probe: Probe) -> ProbeResult:
        raise NotImplementedError('Implement answerProbe(probe): return answer plus tiered requests; never accept gold records')

    def getRequestTelemetry(self) -> list[ModelRequest]:
        raise NotImplementedError('Implement getRequestTelemetry(): return an independent snapshot of every assembled request')


def create_adapter():
    return Adapter()
