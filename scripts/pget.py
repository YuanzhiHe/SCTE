"""Parallel chunked HTTP downloader with resume (for slow single-stream hosts)."""
import os, sys, time, threading
import requests

URL = sys.argv[1]
OUT = sys.argv[2]
NCONN = int(sys.argv[3]) if len(sys.argv) > 3 else 16
CHUNK = 32 * 1024 * 1024

sess_local = threading.local()


def sess():
    if not hasattr(sess_local, "s"):
        sess_local.s = requests.Session()
    return sess_local.s


def head_size(url):
    r = requests.get(url, stream=True, allow_redirects=True, timeout=60)
    r.close()
    return int(r.headers["Content-Length"]), r.url


total, real_url = head_size(URL)
print(f"size {total/1e9:.2f} GB -> {OUT}", flush=True)

done_path = OUT + ".done"
done = set()
if os.path.exists(done_path):
    done = {int(l) for l in open(done_path) if l.strip()}
if not os.path.exists(OUT) or os.path.getsize(OUT) != total:
    with open(OUT, "wb") as f:
        f.truncate(total)
    if not done:
        pass

chunks = [(i, o, min(o + CHUNK, total) - 1)
          for i, o in enumerate(range(0, total, CHUNK)) if i not in done]
print(f"{len(chunks)} chunks left of {(total + CHUNK - 1)//CHUNK}", flush=True)

lock = threading.Lock()
fd = os.open(OUT, os.O_WRONLY)
bytes_done = [0]
t0 = time.time()
donef = open(done_path, "a")


def worker(job):
    idx, start, end = job
    for attempt in range(8):
        try:
            r = sess().get(real_url, headers={"Range": f"bytes={start}-{end}"},
                           stream=True, timeout=120)
            r.raise_for_status()
            buf = r.content
            if len(buf) != end - start + 1:
                raise IOError(f"short chunk {len(buf)}")
            with lock:
                os.pwrite(fd, buf, start)
                donef.write(f"{idx}\n"); donef.flush()
                bytes_done[0] += len(buf)
                el = time.time() - t0
                pct = 100.0 * (len(done) + bytes_done[0] / CHUNK) / ((total + CHUNK - 1) // CHUNK)
                print(f"[{pct:5.1f}%] chunk {idx} ok  {bytes_done[0]/1e6:.0f} MB  "
                      f"{bytes_done[0]/1e6/max(el,1):.2f} MB/s", flush=True)
            return True
        except Exception as e:
            print(f"chunk {idx} retry {attempt}: {e}", flush=True)
            time.sleep(2 + attempt * 3)
    print(f"chunk {idx} FAILED", flush=True)
    return False


from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=NCONN) as ex:
    ok = list(ex.map(worker, chunks))
os.close(fd)
print("ALL_OK" if all(ok) else "SOME_FAILED", flush=True)
