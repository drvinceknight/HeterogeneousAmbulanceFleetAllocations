import subprocess

def test_run_experiment():
    args = ("cd", "..", ";", "python", "src/experiment.py", "10", "10", "240", "40", "500", "6", "0.25", "13", "60", "125", "62", "False")
    result = subprocess.run(args)
    assert result.returncode == 0
