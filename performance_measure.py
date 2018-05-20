import time

def measure_time(func):
    def function_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print("Executing %r takes: %f ms from %f to %f" % (func.__name__, (end_time - start_time) * 1000, start_time, end_time))
        return result
    return function_wrapper

