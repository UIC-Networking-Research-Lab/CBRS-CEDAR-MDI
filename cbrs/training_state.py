"""Completion and early-stopping rules, independent of model/framework imports."""


def completed(checkpoint, target_epochs):
    # A best checkpoint's epoch does not describe how long training continued.
    return (checkpoint.get('training_complete') is True
            and checkpoint.get('target_epochs') == target_epochs)


def advance(score, best, stale, min_delta=1e-6):
    if score > best + min_delta:
        return score, 0
    return best, stale + 1


def device_name(value):
    if value is None:
        return None
    value = str(value)
    return 'cuda:'+value if value.isdigit() else value
