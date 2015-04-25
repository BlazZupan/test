import joblib


def rev(x):
    return -x

parallelizer = joblib.Parallel(n_jobs=-1, max_nbytes=1e3,
                               verbose=0, backend="multiprocessing")
tasks = (joblib.delayed(rev)(x) for x in range(10))
entries = parallelizer(tasks)
