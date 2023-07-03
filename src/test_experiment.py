import subprocess


def test_run_experiment():
    """
    Goes up to root dir and runs command. Captures stdout and confirms exit code
    is 0.

    This does not test the output, just tests that the script runs.
    """
    args = [
        "python",
        "src/experiment.py",
        "10",
        "3",
        "1",
        "1",
        "2",
        "1",
        "5",
        "6",
        "0.25",
        "13",
        "33333",
        "1",
    ]
    result = subprocess.check_call(args, cwd="../")
    assert result == 0
