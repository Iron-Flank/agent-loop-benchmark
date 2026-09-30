"""Operational failures are never incorrect answers."""


class HarnessFailure(Exception):
    category = 'adapter'


class BudgetFailure(HarnessFailure):
    category = 'budget'


class ProviderFailure(HarnessFailure):
    category = 'provider'


class AdapterFailure(HarnessFailure):
    category = 'adapter'


class TimeoutFailure(HarnessFailure):
    category = 'timeout'
