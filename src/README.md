# Development

## Running tests:

```bash
$ cd src/
$ python -m pytest .
```

## Saving a benchmark:

```bash
$ python -m pytest . --benchmark-autosave
```

This saves the benchmarks to an automatically named file. Information on saving
to a named file is available here:
[https://pytest-benchmark.readthedocs.io/en/latest/comparing.html](https://pytest-benchmark.readthedocs.io/en/latest/comparing.html).

## Comparing benchmarks

```bash
$ python -m pytest . --benchmark-compare
```

Information on comparing to a specifically named benchmark is available here:
[https://pytest-benchmark.readthedocs.io/en/latest/comparing.html](https://pytest-benchmark.readthedocs.io/en/latest/comparing.html).

## Not running benchmarks

To only run benchmarked functions once:

```bash
$ python -m pytest . --benchmark-disable
```

Note that this is how continuous integration is configured.
