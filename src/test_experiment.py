import subprocess


def test_run_experiment():
    """
    Goes up to root dir and runs command. Captures stdout and confirms exit code
    is 0.

    This does not test the output, just tests that the script runs.
    """
    args = (
        "cd",
        "..",
        ";",
        "python",
        "src/experiment.py",
        "10",
        "10",
        "240",
        "40",
        "500",
        "6",
        "0.25",
        "13",
        "60",
        "125",
        "62",
        "False",
    )
    result = subprocess.run(args)
    assert result.returncode == 0
